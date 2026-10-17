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
             hidden_size=8, intermediate_size=8, num_experts=4, top_k=2,
             norm_topk_prob=False, rms_norm_eps=1e-5, model_dtype='bfloat16',
             checkpoint_dtype='bfloat16', experts_implementation='eager',
             transformers_version='5.12.1', torch_version=torch.__version__, provenance={})
    write_json(root / 'config.json', c)
    gen = torch.Generator().manual_seed(4)
    for name, shape in {'o_proj': (8, 8), 'post_attention_layernorm': (8,),
                        'router': (4, 8), 'gate_up_proj': (4, 16, 8), 'down_proj': (4, 8, 8)}.items():
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


def test_grouped_reference_matches_actual_module(tiny_export, tmp_path):
    from reference.backend_pairs import dump_backend
    manifest = dump_backend(tiny_export, tmp_path/'grouped', 'float32', 'cpu', 'grouped_mm')
    assert manifest['numerics']['experts_implementation'] == 'grouped_mm'
    assert manifest['numerics']['mirror_equals_module']
    assert manifest['numerics']['grouped_dispatch'] in ('torch.nn.functional.grouped_mm', 'torch._grouped_mm', 'transformers grouped_mm fallback')


def test_mps_fallback_is_rejected(tiny_export, monkeypatch):
    from reference.backend_pairs import compute_backend
    c,a=load_export(tiny_export)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(RuntimeError,match='fallback'):
        compute_backend(c,a,torch.float32,'mps','eager')


@pytest.mark.parametrize('implementation', ['eager','grouped_mm'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16])
def test_backend_forward_hook_null_patch(tiny_export,tmp_path,implementation,dtype):
    from reference.backend_pairs import compute_backend
    c,arrays=load_export(tiny_export)
    baseline,_=compute_backend(c,arrays,dtype,'cpu',implementation)
    patches={}
    for name,value in baseline.items():
        path=tmp_path/f'{name}.npy';save_array(path,value,integer=name=='topk_ids');patches[name]=path
    hooked,_=compute_backend(c,arrays,dtype,'cpu',implementation,patches,patches['topk_ids'])
    assert all(torch.equal(baseline[name],hooked[name]) for name in OPS)


def test_backend_hook_recomputes_downstream(tiny_export,tmp_path):
    from reference.backend_pairs import compute_backend
    c,a=load_export(tiny_export)
    baseline,_=compute_backend(c,a,torch.float32,'cpu','grouped_mm')
    path=tmp_path/'gate.npy';save_array(path,baseline['expert_gate']+1)
    ids=tmp_path/'ids.npy';save_array(ids,baseline['topk_ids'],integer=True)
    patched,_=compute_backend(c,a,torch.float32,'cpu','grouped_mm',{'expert_gate':path},ids)
    assert torch.equal(patched['router_logits'],baseline['router_logits'])
    assert not torch.equal(patched['expert_act'],baseline['expert_act'])
    assert not torch.equal(patched['output'],baseline['output'])


def test_qwen_rectangular_reference(tiny_export):
    from reference.dump_block import compute_ops
    c,a=load_export(tiny_export);c.update(model_type='qwen3_moe',o_proj_input_size=12,norm_topk_prob=True)
    a['o_proj']=np.resize(a['o_proj'],(8,12)).copy();a['o_proj_in']=np.resize(a['o_proj_in'],(5,12)).copy()
    ops=compute_ops(c,a,torch.bfloat16)
    assert ops['output'].shape==(5,8)


def test_qwen_download_plan_selects_only_layer0():
    from reference.export_qwen import required_keys
    keys,shards=required_keys({'model.embed_tokens.weight':'one','model.layers.0.mlp.gate.weight':'one',
                               'model.layers.1.mlp.gate.weight':'two','model.norm.weight':'last'})
    assert len(keys)==2 and shards==['one']


def test_slot_patch_aligns_expert_ids(tiny_export,tmp_path):
    from reference.backend_pairs import compute_backend
    c,a=load_export(tiny_export)
    baseline,_=compute_backend(c,a,torch.float32,'cpu','eager')
    path=tmp_path/'gate.npy';save_array(path,baseline['expert_gate'].flip(1))
    ids=tmp_path/'ids.npy';save_array(ids,baseline['topk_ids'].flip(1),integer=True)
    patched,_=compute_backend(c,a,torch.float32,'cpu','eager',{'expert_gate':path},ids)
    assert torch.equal(patched['expert_gate'],baseline['expert_gate'])
    assert torch.equal(patched['output'],baseline['output'])


def test_nccl_ulp_metric_including_zero():
    from reference.nccl_worker import max_ulp
    ref=np.array([1.],dtype=np.float32)
    assert max_ulp(np.nextafter(ref,np.float32(2)),ref,'fp32')==1
    assert max_ulp(np.array([1+2**-7],dtype=np.float32),ref,'bf16')==1
    assert max_ulp(np.array([2**-133],dtype=np.float32),np.zeros(1,dtype=np.float32),'bf16')==1


def test_rerun_cli_accepts_out_and_patch(tiny_export,tmp_path,monkeypatch):
    import sys
    from reference.backend_pairs import main,compute_backend
    c,a=load_export(tiny_export)
    baseline,_=compute_backend(c,a,torch.float32,'cpu','eager')
    patch=tmp_path/'output.npy';save_array(patch,baseline['output']+1)
    dest=tmp_path/'rerun'
    monkeypatch.setattr(sys,'argv',['backend_pairs','--export',str(tiny_export),'--out',str(dest),
                                    '--dtype','float32','--implementation','eager','--patch',f'output={patch}'])
    main()
    assert np.array_equal(np.load(dest/'output.npy'),(baseline['output']+1).numpy())
    assert not json.loads((dest/'manifest.json').read_text())['numerics']['mirror_equals_module']


def test_known_demo_hook_is_removed_by_patch(tiny_export,tmp_path):
    from reference.backend_pairs import compute_backend
    c,a=load_export(tiny_export)
    clean,_=compute_backend(c,a,torch.bfloat16,'cpu','eager')
    faulty,_=compute_backend(c,a,torch.bfloat16,'cpu','eager',demo_scale_combine=16.)
    assert not torch.equal(faulty['output'],clean['output'])
    patch=tmp_path/'combine.npy';save_array(patch,clean['combine_terms'])
    ids=tmp_path/'ids.npy';save_array(ids,clean['topk_ids'],integer=True)
    repaired,_=compute_backend(c,a,torch.bfloat16,'cpu','eager',{'combine_terms':patch},ids,16.)
    assert all(torch.equal(repaired[name],clean[name]) for name in OPS)


def test_stride_sensitive_activation_is_traced_and_fixed(tiny_export,tmp_path,monkeypatch):
    """Controlled dispatch simulation, not evidence about a particular x86 CPU."""
    from transformers.activations import SiLUActivation
    from reference.backend_pairs import compute_backend
    from reference.mirror_diagnostics import MirrorMismatch
    original=SiLUActivation.forward
    def stride_sensitive(self,value):
        result=original(self,value)
        return result+0.01 if value.stride(0)==value.shape[-1] else result
    monkeypatch.setattr(SiLUActivation,'forward',stride_sensitive)
    monkeypatch.setenv('COINFLIP_MIRROR_DIAGNOSTICS',str(tmp_path/'diagnostics'))
    c,arrays=load_export(tiny_export)
    with pytest.raises(MirrorMismatch,match='first op=expert_act'):
        compute_backend(c,arrays,torch.float32,'cpu','eager',_legacy_expert_layout=True)
    report=next((tmp_path/'diagnostics').glob('*/diagnostic.json'))
    result=json.loads(report.read_text())
    assert result['first_differing_op']=='expert_act'
    assert result['ops']['expert_gate']['bit_exact']
    assert (report.parent/'module/expert_act.npy').is_file()
    assert (report.parent/'mirror/expert_act.npy').is_file()
    # The corrected mirror remains strict and passes the same simulated dispatch.
    compute_backend(c,arrays,torch.float32,'cpu','eager')


def test_actual_eager_trace_matches_reference(tiny_export):
    from reference.mirror_diagnostics import trace_eager,describe
    c,arrays=load_export(tiny_export)
    c['norm_topk_prob']=True
    baseline=compute_ops(c,arrays,torch.bfloat16)
    module,_=make_modules(c,arrays,torch.bfloat16)
    actual,_=trace_eager(module,baseline['router_input'],baseline)
    assert describe(baseline,actual)['first_differing_op'] is None


def test_pre_routing_demo_changes_routes_and_patch_repairs(tiny_export,tmp_path):
    from reference.backend_pairs import compute_backend
    c,a=load_export(tiny_export)
    clean,_=compute_backend(c,a,torch.bfloat16,'cpu','eager')
    faulty,_=compute_backend(c,a,torch.bfloat16,'cpu','eager',demo_scale_router=-1.)
    assert not torch.equal(torch.sort(faulty['topk_ids']).values,torch.sort(clean['topk_ids']).values)
    patch=tmp_path/'logits.npy';save_array(patch,clean['router_logits'])
    repaired,_=compute_backend(c,a,torch.bfloat16,'cpu','eager',{'router_logits':patch},demo_scale_router=-1.)
    assert all(torch.equal(clean[op],repaired[op]) for op in OPS)
