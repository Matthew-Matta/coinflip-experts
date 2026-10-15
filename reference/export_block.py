"""Capture real prefixes and export OLMoE blocks without loading the full model.

A local JSONL pool supplies {id, source, text}; source is one of GSM8K, HumanEval,
ShareGPT, FLORES. Selection is seeded and saved before inference. Use
reference.select_prompts to obtain this pool from revision-pinned datasets.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import random
import re
import resource
import sys
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer, OlmoeModel
from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer

from coinflip.provenance import collect
from reference.dump_block import require_version
from reference.format import save_array, sha256, write_json

MODEL_ID = 'allenai/OLMoE-1B-7B-0924'
SOURCES = ('GSM8K', 'HumanEval', 'ShareGPT', 'FLORES')


def select_prompts(pool: list[dict], tokenizer, seed: int = 42, per_source: int = 8,
                   max_tokens: int = 64) -> list[dict]:
    if per_source < 2 or per_source % 2 or max_tokens < 1:
        raise ValueError('per_source must be positive and even (>=2); max_tokens must be positive')
    rng = random.Random(seed)
    selected = []
    for source in SOURCES:
        rows = [dict(r) for r in pool if r['source'] == source]
        ids = [str(r['id']) for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f'duplicate prompt IDs in {source}')
        if len(rows) < per_source:
            raise ValueError(f'{source}: need {per_source} prompts, have {len(rows)}')
        for n, row in enumerate(rng.sample(rows, per_source)):
            row['prompt_id'] = f'{source}:{row["id"]}'
            row['dataset_split'] = row.get('split')
            row['split'] = 'calibration' if n < per_source // 2 else 'evaluation'
            row['token_ids'] = tokenizer(row['text'], add_special_tokens=True,
                                         truncation=True, max_length=max_tokens)['input_ids']
            if not row['token_ids']:
                raise ValueError(f'empty prompt: {row["prompt_id"]}')
            selected.append(row)
    return selected


def estimated_model_bytes(cfg, layers: int) -> int:
    """Conservative weight estimate for truncated bf16 OlmoeModel (no lm_head)."""
    h, i, e = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts
    attn = 2*h*h + 2*h*cfg.num_key_value_heads*(h // cfg.num_attention_heads)
    return 2 * (cfg.vocab_size*h + layers*(3*e*i*h + e*h + attn + 4*h) + h)


def peak_rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**3 if sys.platform == 'darwin' else 1024**2)


@torch.inference_mode()
def capture_and_export(model, selected: list[dict], layers: list[int], output: Path,
                       revision: str, *, model_id: str = MODEL_ID) -> list[Path]:
    """Capture layer-entry residuals and o_proj pre-hook inputs on identical tokens."""
    cfg = model.config
    captured = {layer: {} for layer in layers}
    width = max(len(row['token_ids']) for row in selected)
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    ids = torch.full((len(selected), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    meta = []
    for b, row in enumerate(selected):
        length = len(row['token_ids'])
        ids[b, :length] = torch.tensor(row['token_ids'])
        mask[b, :length] = 1
        meta.extend(dict(prompt_id=row['prompt_id'], position=p, split=row['split']) for p in range(length))
    valid = mask.bool()
    handles = []
    def capture(layer, name):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs['hidden_states']
            captured[layer][name] = value[valid].detach().clone()
        return hook
    try:
        for layer in layers:
            decoder = model.layers[layer]
            handles.append(decoder.register_forward_pre_hook(capture(layer, 'residual'), with_kwargs=True))
            handles.append(decoder.self_attn.o_proj.register_forward_pre_hook(capture(layer, 'o_proj_in'), with_kwargs=True))
        start = time.perf_counter()
        model(input_ids=ids, attention_mask=mask, use_cache=False)
        elapsed = time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
    paths = []
    for layer in layers:
        directory = output / f'layer{layer}'
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'refusing to overwrite {directory}')
        (directory / 'weights').mkdir(parents=True)
        (directory / 'inputs').mkdir()
        decoder = model.layers[layer]
        weights = dict(o_proj=decoder.self_attn.o_proj.weight,
                       post_attention_layernorm=decoder.post_attention_layernorm.weight,
                       router=decoder.mlp.gate.weight,
                       gate_up_proj=decoder.mlp.experts.gate_up_proj,
                       down_proj=decoder.mlp.experts.down_proj)
        for name, value in weights.items():
            save_array(directory / 'weights' / f'{name}.npy', value)
        for name, value in captured[layer].items():
            save_array(directory / 'inputs' / f'{name}.npy', value)
        save_array(directory / 'inputs/token_ids.npy', ids[valid], integer=True)
        write_json(directory / 'inputs/token_meta.json', meta)
        source = Path(inspect.getsourcefile(OlmoeDecoderLayer))
        config = dict(model_id=model_id, hf_revision=revision, layer=layer,
                      hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
                      num_experts=cfg.num_experts, top_k=cfg.num_experts_per_tok,
                      norm_topk_prob=cfg.norm_topk_prob, rms_norm_eps=cfg.rms_norm_eps,
                      model_dtype='bfloat16', checkpoint_dtype='bfloat16',
                      transformers_version=transformers.__version__, experts_implementation='eager',
                      torch_version=torch.__version__,
                      capture=dict(batch_size=len(selected), padded_tokens=width,
                                   valid_tokens=len(meta), wall_seconds=elapsed, peak_rss_gib=peak_rss_gib(),
                                   transformers_source_sha256=sha256(source)),
                      provenance=collect({'layer': layer, 'revision': revision}))
        write_json(directory / 'config.json', config)
        print(f'PASS export layer={layer} tokens={len(meta)} capture_s={elapsed:.3f} peak_rss_GiB={peak_rss_gib():.3f}', flush=True)
        paths.append(directory)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--revision', default='main')
    p.add_argument('--layers', type=int, nargs='+', default=[0, 7, 15])
    p.add_argument('--out', type=Path, default=Path('data/export/olmoe-1b-7b-0924'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--per-source', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=64)
    p.add_argument('--max-model-gib', type=float, default=8.0,
                   help='weight-only cap; reserve remaining RAM for export and activations')
    p.add_argument('--cache-dir', type=Path, default=Path('data/hf-cache/hub'))
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    require_version()
    torch.set_num_threads(a.threads)
    revision = a.revision if re.fullmatch(r'[0-9a-f]{40}', a.revision) else HfApi().model_info(a.model, revision=a.revision).sha
    cfg = AutoConfig.from_pretrained(a.model, revision=revision, cache_dir=a.cache_dir,
                                     local_files_only=a.local_files_only)
    if cfg.dtype != torch.bfloat16:
        raise ValueError(f'expected a bf16 checkpoint, got {cfg.dtype}; refusing to mislabel converted weights')
    requested = sorted(set(a.layers))
    if not requested or requested[0] < 0 or requested[-1] >= cfg.num_hidden_layers:
        p.error('layers outside checkpoint')
    allowed = [n for n in requested if estimated_model_bytes(cfg, n+1) <= a.max_model_gib*1024**3]
    skipped = sorted(set(requested) - set(allowed))
    if not allowed:
        raise RuntimeError('no requested layers fit the weight budget')
    for n in skipped:
        print(f'SKIP layer={n}: estimated weights={estimated_model_bytes(cfg,n+1)/1024**3:.2f} GiB exceed cap; use a high-memory CPU job', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=revision, cache_dir=a.cache_dir,
                                              local_files_only=a.local_files_only)
    tokenizer.padding_side = 'right'
    with a.pool.open() as stream:
        pool = [json.loads(line) for line in stream if line.strip()]
    selected = select_prompts(pool, tokenizer, a.seed, a.per_source, a.max_tokens)
    a.out.mkdir(parents=True, exist_ok=True)
    selection_path = a.out / 'selection.json'
    if selection_path.exists():
        raise FileExistsError(f'refusing to overwrite {selection_path}')
    write_json(selection_path, {'seed': a.seed, 'pool_sha256': sha256(a.pool), 'hf_revision': revision, 'prompts': selected})
    cfg.num_hidden_layers = max(allowed)+1
    cfg.use_cache = False
    start = time.perf_counter()
    model, loading = OlmoeModel.from_pretrained(a.model, config=cfg, revision=revision, cache_dir=a.cache_dir,
                                       local_files_only=a.local_files_only,
                                       dtype=torch.bfloat16, experts_implementation='eager',
                                       attn_implementation='eager', output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'checkpoint did not completely initialise the truncated model: {loading}')
    model.eval()
    print(f'PASS loaded {len(model.layers)} layers in {time.perf_counter()-start:.2f}s; peak_rss_GiB={peak_rss_gib():.3f}', flush=True)
    paths = capture_and_export(model, selected, allowed, a.out, revision, model_id=a.model)
    del model
    gc.collect()
    write_json(a.out / 'capture_summary.json', dict(exports=[str(x) for x in paths], skipped_layers=skipped,
               peak_rss_gib=peak_rss_gib(), load_and_capture_seconds=time.perf_counter()-start))


if __name__ == '__main__':
    main()
