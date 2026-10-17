"""Dump the 13 contracted operations; validate the mirror against Transformers eager.

    python -m reference.dump_block --export data/export/.../layer0 --dtype bfloat16 --dump data/dumps/torch-bf16
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm, OlmoeSparseMoeBlock

from coinflip.provenance import collect
from reference.format import OPS, expected_shapes, load_export, save_array, sha256, write_json

TRANSFORMERS_VERSION = '5.12.1'
DTYPES = {'bfloat16': torch.bfloat16, 'float32': torch.float32}


def require_version() -> None:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f'requires transformers=={TRANSFORMERS_VERSION}, got {transformers.__version__}')


def make_modules(c: dict, arrays: dict, dtype: torch.dtype, *, implementation='eager', device='cpu'):
    if c.get('model_type') == 'qwen3_moe':
        from transformers import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm, Qwen3MoeSparseMoeBlock
        cfg = Qwen3MoeConfig(hidden_size=c['hidden_size'], moe_intermediate_size=c['intermediate_size'],
                            num_experts=c['num_experts'], num_experts_per_tok=c['top_k'],
                            norm_topk_prob=c['norm_topk_prob'], rms_norm_eps=c['rms_norm_eps'])
        block_class, norm_class = Qwen3MoeSparseMoeBlock, Qwen3MoeRMSNorm
    else:
        block_class, norm_class = OlmoeSparseMoeBlock, OlmoeRMSNorm
        cfg = OlmoeConfig(hidden_size=c['hidden_size'], intermediate_size=c['intermediate_size'],
                          num_experts=c['num_experts'], num_experts_per_tok=c['top_k'],
                          norm_topk_prob=c['norm_topk_prob'], rms_norm_eps=c['rms_norm_eps'],
                          hidden_act='silu')
    cfg._experts_implementation = implementation
    # Allocate directly in the model dtype, avoiding a large temporary fp32 block.
    with torch.device('meta'):
        module = block_class(cfg)
        norm = norm_class(c['hidden_size'], eps=c['rms_norm_eps'])
    tensor = lambda name: torch.from_numpy(arrays[name]).to(device=device, dtype=dtype)
    module.load_state_dict({
        'gate.weight': tensor('router'),
        'experts.gate_up_proj': tensor('gate_up_proj'),
        'experts.down_proj': tensor('down_proj'),
    }, assign=True)
    norm.load_state_dict({'weight': tensor('post_attention_layernorm')}, assign=True)
    return module.eval(), norm.eval()


@torch.inference_mode()
def compute_ops(c: dict, arrays: dict, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Mirror the pinned eager implementation, including expert-order index_add_."""
    module, norm = make_modules(c, arrays, dtype)
    tensor = lambda name: torch.from_numpy(arrays[name]).to(dtype=dtype)
    ops = {}
    ops['o_proj_out'] = F.linear(tensor('o_proj_in'), tensor('o_proj'))
    ops['resid_sum'] = tensor('residual') + ops['o_proj_out']
    x = ops['router_input'] = norm(ops['resid_sum'])
    ops['router_logits'] = F.linear(x, module.gate.weight)
    ops['router_probs'] = F.softmax(ops['router_logits'], dim=-1, dtype=torch.float32)
    weights, ids = torch.topk(ops['router_probs'], c['top_k'], dim=-1)
    if c['norm_topk_prob']:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    ops['topk_ids'] = ids.to(torch.int32)
    ops['topk_weights'] = weights.to(dtype)
    t, k, i, h = len(x), c['top_k'], c['intermediate_size'], c['hidden_size']
    for name, width in (('expert_gate', i), ('expert_up', i), ('expert_act', i),
                        ('expert_down', h), ('combine_terms', h)):
        ops[name] = torch.empty((t, k, width), dtype=dtype)
    out = torch.zeros_like(x)
    # Match modeling_olmoe.py:321-336, including slot-major indexing within each expert.
    mask = F.one_hot(ids, num_classes=c['num_experts']).permute(2, 1, 0)
    for expert in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert = expert[0]
        slots, tokens = torch.where(mask[expert])
        gate, up = F.linear(x[tokens], module.experts.gate_up_proj[expert]).chunk(2, dim=-1)
        act = module.experts.act_fn(gate) * up
        down = F.linear(act, module.experts.down_proj[expert])
        terms = down * ops['topk_weights'][tokens, slots, None]
        for name, values in (('expert_gate', gate), ('expert_up', up), ('expert_act', act),
                             ('expert_down', down), ('combine_terms', terms)):
            ops[name][tokens, slots] = values
        out.index_add_(0, tokens, terms.to(out.dtype))
    ops['output'] = out
    actual = module(x.unsqueeze(0)).squeeze(0)
    if not torch.equal(out, actual):
        err = (out.float() - actual.float()).abs().max().item()
        raise RuntimeError(f'mirror != Transformers module: max absolute error {err}; refusing reference dump')
    for name, shape in expected_shapes(c, t).items():
        if tuple(ops[name].shape) != shape or not torch.isfinite(ops[name]).all():
            raise ValueError(f'invalid or nonfinite op {name}')
    return ops


def dump_block(export: Path, destination: Path, dtype_name: str = 'bfloat16') -> dict:
    require_version()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty dump: {destination}')
    c, arrays = load_export(export)
    ops = compute_ops(c, arrays, DTYPES[dtype_name])
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in OPS:
        integer = name == 'topk_ids'
        save_array(destination / f'{name}.npy', ops[name], integer=integer)
        entries.append({'name': name, 'file': f'{name}.npy', 'shape': list(ops[name].shape),
                        'logical_dtype': 'int32' if integer else ('fp32' if name == 'router_probs' or dtype_name == 'float32' else 'bf16'),
                        'stage': 'selection' if OPS.index(name) < 6 else 'post-selection'})
    source_class = OlmoeSparseMoeBlock
    if c.get('model_type') == 'qwen3_moe':
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        source_class = Qwen3MoeSparseMoeBlock
    source = Path(inspect.getsourcefile(source_class))
    manifest = {
        'run_id': destination.name, 'backend': 'torch-cpu',
        'export': {'path': str(export), 'config_sha256': sha256(export / 'config.json')},
        'numerics': {'dtype': dtype_name, 'experts_implementation': 'eager',
                     'mirror_equals_module': True, 'transformers_source_sha256': sha256(source),
                     'torch_num_threads': torch.get_num_threads()},
        'bug': None, 'tokens': len(arrays['token_ids']), 'ops': entries,
        'provenance': collect({'dtype': dtype_name, 'layer': c['layer']}),
    }
    write_json(destination / 'manifest.json', manifest)
    print(f'PASS mirror == Transformers eager bit-exactly; {len(ops)} ops, {manifest["tokens"]} tokens, {dtype_name}')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--dump', type=Path, required=True)
    p.add_argument('--dtype', choices=DTYPES, default='bfloat16')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.threads < 1:
        p.error('--threads must be positive')
    torch.set_num_threads(a.threads)
    dump_block(a.export, a.dump, a.dtype)


if __name__ == '__main__':
    main()
