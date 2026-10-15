import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from reference.dump_block import compute_ops, dump_block, make_modules
from reference.format import OPS, expected_shapes, load_export, save_array, write_json


@pytest.fixture
def tiny_export(tmp_path):
    root = tmp_path / 'layer0'
    (root / 'weights').mkdir(parents=True)
    (root / 'inputs').mkdir()
    c = dict(model_id='synthetic-test-fixture', hf_revision='fixture', layer=0,
             hidden_size=8, intermediate_size=4, num_experts=4, top_k=2,
             norm_topk_prob=False, rms_norm_eps=1e-5, model_dtype='bfloat16',
             checkpoint_dtype='bfloat16', experts_implementation='eager',
             transformers_version='5.12.1', torch_version=torch.__version__, provenance={})
    write_json(root / 'config.json', c)
    gen = torch.Generator().manual_seed(4)
    for name, shape in {'o_proj': (8, 8), 'post_attention_layernorm': (8,),
                        'router': (4, 8), 'gate_up_proj': (4, 8, 8), 'down_proj': (4, 8, 4)}.items():
        value = (torch.randn(shape, generator=gen) * 0.2).bfloat16()
        save_array(root / 'weights' / f'{name}.npy', value)
    for name in ('residual', 'o_proj_in'):
        save_array(root / 'inputs' / f'{name}.npy', torch.randn(5, 8, generator=gen).bfloat16())
    save_array(root / 'inputs/token_ids.npy', np.arange(5), integer=True)
    write_json(root / 'inputs/token_meta.json', [dict(prompt_id='p0', position=i, split='calibration') for i in range(5)])
    return root


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
def test_dump_contract_and_mirror(tiny_export, tmp_path, dtype):
    dest = tmp_path / dtype
    manifest = dump_block(tiny_export, dest, dtype)
    c, arrays = load_export(tiny_export)
    assert len(manifest['ops']) == 13
    assert [o['name'] for o in manifest['ops']] == list(OPS)
    assert manifest['numerics']['mirror_equals_module'] is True
    assert manifest['bug'] is None
    assert json.loads((dest / 'manifest.json').read_text()) == manifest
    for op in manifest['ops']:
        a = np.load(dest / op['file'], allow_pickle=False)
        assert a.shape == expected_shapes(c, 5)[op['name']]
        assert a.dtype == np.dtype('<i4' if op['name'] == 'topk_ids' else '<f4')
        if op['logical_dtype'] == 'bf16':
            assert torch.equal(torch.from_numpy(a), torch.from_numpy(a).bfloat16().float())
    ids = np.load(dest / 'topk_ids.npy')
    probs = np.load(dest / 'router_probs.npy')
    assert np.all(np.take_along_axis(probs, ids[:, :1], -1) >= np.take_along_axis(probs, ids[:, 1:], -1))
    np.testing.assert_allclose(probs.sum(-1), 1, atol=2e-7)
    with pytest.raises(FileExistsError):
        dump_block(tiny_export, dest, dtype)


def test_mirror_failure_refuses_dump(tiny_export, tmp_path, monkeypatch):
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    original = OlmoeSparseMoeBlock.forward
    monkeypatch.setattr(OlmoeSparseMoeBlock, 'forward', lambda self, x: original(self, x) + 1)
    with pytest.raises(RuntimeError, match='mirror !='):
        dump_block(tiny_export, tmp_path / 'bad')
    assert not (tmp_path / 'bad/manifest.json').exists()


def test_renormalized_topk(tiny_export):
    c, arrays = load_export(tiny_export)
    c['norm_topk_prob'] = True
    ops = compute_ops(c, arrays, torch.float32)
    torch.testing.assert_close(ops['topk_weights'].sum(-1), torch.ones(5))


def test_bad_export_rejected(tiny_export):
    save_array(tiny_export / 'inputs/residual.npy', np.zeros((2, 8)))
    with pytest.raises(ValueError, match='residual'):
        load_export(tiny_export)


def test_capture_real_module_hooks(tmp_path):
    from transformers import OlmoeConfig, OlmoeModel
    from reference.export_block import capture_and_export
    cfg = OlmoeConfig(hidden_size=8, intermediate_size=4, num_experts=4, num_experts_per_tok=2,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                      vocab_size=40, pad_token_id=0)
    cfg._experts_implementation = 'eager'
    cfg._attn_implementation = 'eager'
    torch.manual_seed(2)
    model = OlmoeModel(cfg).to(torch.bfloat16).eval()
    prompts = [dict(prompt_id='a', split='calibration', token_ids=[2, 3, 4]),
               dict(prompt_id='b', split='evaluation', token_ids=[5, 6])]
    expected = []
    handle = model.layers[0].mlp.register_forward_pre_hook(lambda m, args: expected.append(args[0].detach().clone()))
    roots = capture_and_export(model, prompts, [0], tmp_path / 'export', 'fixture', model_id='synthetic-test-fixture')
    handle.remove()
    c, arrays = load_export(roots[0])
    ops = compute_ops(c, arrays, torch.bfloat16)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    assert torch.equal(ops['router_input'], expected[0][valid])
    meta = json.loads((roots[0] / 'inputs/token_meta.json').read_text())
    assert [r['prompt_id'] for r in meta] == ['a', 'a', 'a', 'b', 'b']
    assert [r['position'] for r in meta] == [0, 1, 2, 0, 1]


def test_seeded_prompt_split():
    from reference.export_block import select_prompts, SOURCES
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return dict(input_ids=list(range(min(len(text), kwargs['max_length']))))
    pool = [dict(id=str(i), source=s, text='example prompt') for s in SOURCES for i in range(12)]
    a = select_prompts(pool, Tokenizer())
    assert a == select_prompts(pool, Tokenizer())
    assert len(a) == 32
    assert sum(r['split'] == 'calibration' for r in a) == 16
    for source in SOURCES:
        rows = [r for r in a if r['source'] == source]
        assert len(rows) == 8
        assert len({r['prompt_id'] for r in rows}) == 8


def test_sharegpt_column_formats():
    from reference.select_prompts import first_human_turn
    messages = [dict(**{'from': 'human', 'value': 'question'}), dict(**{'from': 'gpt', 'value': 'answer'})]
    assert first_human_turn(messages) == 'question'
    assert first_human_turn(json.dumps(messages)) == 'question'
    assert first_human_turn([json.dumps(m) for m in messages]) == 'question'
    assert first_human_turn({'from': ['human', 'gpt'], 'value': ['question', 'answer']}) == 'question'


def test_incremental_prompt_json():
    from reference.select_prompts import parse_json_array
    text = json.dumps([{'id': 'a', 'conversations': [{'from': 'human', 'value': 'a, b [test]'}]}, {'id': 'b'}])
    assert list(parse_json_array(text[i:i+7] for i in range(0, len(text), 7))) == json.loads(text)
    with pytest.raises(ValueError, match='truncated'):
        list(parse_json_array(['[{"id":']))


def test_original_flores_archive(tmp_path):
    import io
    import tarfile
    from reference.select_prompts import flores_rows
    archive_path = tmp_path / 'flores.tar.gz'
    data = b'First sentence.\nSecond sentence.\n'
    with tarfile.open(archive_path, 'w:gz') as archive:
        info = tarfile.TarInfo('flores200_dataset/devtest/eng_Latn.devtest')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    rows, revision = flores_rows(archive_path)
    assert rows == [{'id': '1', 'sentence': 'First sentence.'}, {'id': '2', 'sentence': 'Second sentence.'}]
    assert revision.startswith('sha256:')
