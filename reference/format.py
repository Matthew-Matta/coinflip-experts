"""Readers and writers for spec/INTERFACES.md (little-endian C-order arrays)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

OPS = (
    'o_proj_out', 'resid_sum', 'router_input', 'router_logits', 'router_probs',
    'topk_ids', 'topk_weights', 'expert_gate', 'expert_up', 'expert_act',
    'expert_down', 'combine_terms', 'output',
)
WEIGHTS = ('o_proj', 'post_attention_layernorm', 'router', 'gate_up_proj', 'down_proj')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_array(path: Path, value, *, integer: bool = False) -> None:
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        value = value.int().numpy() if integer else value.float().numpy()
    arr = np.asarray(value, dtype='<i4' if integer else '<f4', order='C')
    if not integer and not np.isfinite(arr).all():
        raise ValueError(f'nonfinite tensor: {path.name}')
    np.save(path, arr, allow_pickle=False)


def expected_shapes(c: dict, tokens: int) -> dict[str, tuple[int, ...]]:
    h, i, e, k = (c[n] for n in ('hidden_size', 'intermediate_size', 'num_experts', 'top_k'))
    return dict(zip(OPS, (
        (tokens, h), (tokens, h), (tokens, h), (tokens, e), (tokens, e),
        (tokens, k), (tokens, k), (tokens, k, i), (tokens, k, i), (tokens, k, i),
        (tokens, k, h), (tokens, k, h), (tokens, h),
    ), strict=True))


def load_export(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    c = json.loads((directory / 'config.json').read_text())
    h, i, e, k = (c[n] for n in ('hidden_size', 'intermediate_size', 'num_experts', 'top_k'))
    if not all(isinstance(v, int) and v > 0 for v in (h, i, e, k)) or k >= e:
        raise ValueError('invalid export dimensions or top_k')
    if c['experts_implementation'] != 'eager':
        raise ValueError('reference requires experts_implementation=eager')
    shapes = dict(zip(WEIGHTS, ((h, h), (h,), (e, h), (e, 2*i, h), (e, h, i)), strict=True))
    arrays = {}
    for name in WEIGHTS + ('o_proj_in', 'residual', 'token_ids'):
        sub = 'weights' if name in WEIGHTS else 'inputs'
        a = np.load(directory / sub / f'{name}.npy', allow_pickle=False)
        dtype = np.dtype('<i4' if name == 'token_ids' else '<f4')
        if a.dtype != dtype or not a.flags.c_contiguous or not np.isfinite(a).all():
            raise ValueError(f'{name}: expected finite C-order {dtype}')
        if name in shapes and a.shape != shapes[name]:
            raise ValueError(f'{name}: shape {a.shape} != {shapes[name]}')
        arrays[name] = a
    t = len(arrays['token_ids'])
    if t == 0 or arrays['token_ids'].shape != (t,):
        raise ValueError('token_ids must be a nonempty vector')
    for name in ('o_proj_in', 'residual'):
        if arrays[name].shape != (t, h):
            raise ValueError(f'{name}: expected {(t, h)}')
    meta = json.loads((directory / 'inputs/token_meta.json').read_text())
    if len(meta) != t:
        raise ValueError('token metadata length does not match inputs')
    return c, arrays
