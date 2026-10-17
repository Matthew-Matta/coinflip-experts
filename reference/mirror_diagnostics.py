"""Trace the actual eager module without changing tensor layouts or arithmetic."""
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from reference.format import OPS, save_array, write_json


class MirrorMismatch(RuntimeError):
    """Strict parity failure, retained separately from unsupported hardware."""


@torch.inference_mode()
def trace_eager(module, x, mirror):
    traced = {name: mirror[name] for name in OPS[:3]}
    for name in OPS[7:-1]:
        traced[name] = torch.empty_like(mirror[name])
    linear, softmax, topk = F.linear, F.softmax, torch.topk
    gate_ptr = {module.experts.gate_up_proj[e].data_ptr(): e for e in range(module.experts.num_experts)}
    down_ptr = {module.experts.down_proj[e].data_ptr(): e for e in range(module.experts.num_experts)}
    calls = []

    def locations(e):
        mask = F.one_hot(traced['topk_ids'].long(), num_classes=module.experts.num_experts).permute(2, 1, 0)
        return torch.where(mask[e])

    def observed_linear(value, weight, *args, **kwargs):
        result = linear(value, weight, *args, **kwargs)
        ptr = weight.data_ptr()
        if ptr == module.gate.weight.data_ptr():
            traced['router_logits'] = result
        elif ptr in gate_ptr:
            e = gate_ptr[ptr]; slots, tokens = locations(e)
            gate, up = result.chunk(2, dim=-1)
            traced['expert_gate'][tokens, slots] = gate
            traced['expert_up'][tokens, slots] = up
            calls.append(dict(expert=e, rows=len(value), gate_stride=list(gate.stride()),
                              gate_contiguous=gate.is_contiguous()))
        elif ptr in down_ptr:
            e = down_ptr[ptr]; slots, tokens = locations(e)
            traced['expert_act'][tokens, slots] = value
            traced['expert_down'][tokens, slots] = result
            traced['combine_terms'][tokens, slots] = result * traced['topk_weights'][tokens, slots, None]
        return result

    def observed_softmax(*args, **kwargs):
        result = softmax(*args, **kwargs); traced['router_probs'] = result
        return result

    def observed_topk(*args, **kwargs):
        weights, ids = topk(*args, **kwargs)
        traced['topk_ids'] = ids.to(torch.int32)
        # OLMoE normalises these weights in place when configured to do so.
        traced['topk_weights'] = weights
        return weights, ids

    def router_result(_module, _args, result):
        traced['topk_weights'] = result[1]

    handle = module.gate.register_forward_hook(router_result)
    try:
        with patch.object(F, 'linear', observed_linear), patch.object(F, 'softmax', observed_softmax), patch.object(torch, 'topk', observed_topk):
            traced['output'] = module(x.unsqueeze(0)).squeeze(0)
    finally:
        handle.remove()
    traced['topk_weights'] = traced['topk_weights'].to(x.dtype)
    return traced, calls


def describe(mirror, actual):
    metrics = {}
    for name in OPS:
        a = mirror[name].detach().float().cpu().numpy()
        b = actual[name].detach().float().cpu().numpy()
        different = a.view(np.uint32) != b.view(np.uint32)
        metrics[name] = dict(bit_exact=not bool(different.any()), different_elements=int(different.sum()),
                             max_absolute=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
    first = next((name for name in OPS if not metrics[name]['bit_exact']), None)
    return dict(first_differing_op=first, ops=metrics)


def diagnose(module, x, mirror, destination):
    actual, calls = trace_eager(module, x, mirror)
    result = describe(mirror, actual)
    result.update(torch_version=torch.__version__, platform=platform.platform(),
                  threads=torch.get_num_threads(), torch_config=torch.__config__.show(), expert_calls=calls,
                  scope='Actual module and mirror share router_input; upstream ops copied, not independently validated here')
    if destination is not None:
        destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
        for label, values in [('mirror', mirror), ('module', actual)]:
            root = destination / label; root.mkdir(exist_ok=True)
            for name in OPS: save_array(root / f'{name}.npy', values[name], integer=name=='topk_ids')
        write_json(destination/'diagnostic.json', result)
    return result


def check_eager(module, x, mirror):
    actual = module(x.unsqueeze(0)).squeeze(0)
    if torch.equal(mirror['output'], actual): return
    root = os.environ.get('COINFLIP_MIRROR_DIAGNOSTICS')
    if root:
        import uuid
        root = Path(root) / uuid.uuid4().hex[:12]
    details = diagnose(module, x, mirror, root)
    print('MIRROR PARITY FAILURE: '+json.dumps(details), flush=True)
    raise MirrorMismatch(f'eager mirror != actual Transformers module; first op={details["first_differing_op"]}; diagnostics={root}')
