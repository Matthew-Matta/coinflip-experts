"""Real backend dumps: eager on MPS, or Transformers grouped_mm on CPU.

No CPU fallback is enabled for MPS. Unsupported paths produce status.json, not a
successful dump. A grouped_mm dump mirrors the actual grouped implementation,
including its slot-order sum, rather than copying eager intermediate tensors.
"""
from __future__ import annotations
import argparse
import inspect
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers.integrations import moe
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from coinflip.provenance import collect
from reference.dump_block import DTYPES, make_modules, require_version
from reference.format import OPS, expected_shapes, load_export, save_array, sha256, write_json


@torch.inference_mode()
def compute_backend(c, arrays, dtype, device, implementation):
    if device == 'mps':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('MPS CPU fallback is enabled; refusing to mislabel fallback execution')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable in this process')
    module, norm = make_modules(c, arrays, dtype, implementation=implementation, device=device)
    tensor=lambda name: torch.from_numpy(arrays[name]).to(device=device,dtype=dtype)
    ops={}
    ops['o_proj_out']=F.linear(tensor('o_proj_in'),tensor('o_proj'))
    ops['resid_sum']=tensor('residual')+ops['o_proj_out']
    x=ops['router_input']=norm(ops['resid_sum'])
    logits, weights, ids=module.gate(x)
    ops.update(router_logits=logits, router_probs=F.softmax(logits,dim=-1,dtype=torch.float32),
               topk_ids=ids.to(torch.int32), topk_weights=weights)
    t,k,i,h=len(x),c['top_k'],c['intermediate_size'],c['hidden_size']
    dispatch='eager'
    if implementation=='eager':
        for name,width in (('expert_gate',i),('expert_up',i),('expert_act',i),('expert_down',h),('combine_terms',h)):
            ops[name]=torch.empty((t,k,width),dtype=dtype,device=device)
        out=torch.zeros_like(x)
        mask=F.one_hot(ids,num_classes=c['num_experts']).permute(2,1,0)
        for e in torch.greater(mask.sum(dim=(-1,-2)),0).nonzero():
            e=e[0]
            slots,tokens=torch.where(mask[e])
            gate,up=F.linear(x[tokens],module.experts.gate_up_proj[e]).chunk(2,dim=-1)
            act=module.experts.act_fn(gate)*up
            down=F.linear(act,module.experts.down_proj[e])
            terms=down*weights[tokens,slots,None]
            for name,values in (('expert_gate',gate),('expert_up',up),('expert_act',act),('expert_down',down),('combine_terms',terms)):
                ops[name][tokens,slots]=values
            out.index_add_(0,tokens,terms.to(out.dtype))
    else:
        expert_ids,perm=torch.sort(ids.reshape(-1))
        selected=x[perm//k]
        histc_input=expert_ids.float() if device in ('cpu','mps') else expert_ids.int()
        counts=torch.histc(histc_input,bins=c['num_experts'],min=0,max=c['num_experts']-1)
        offsets=torch.cumsum(counts,dim=0,dtype=torch.int32)
        native=moe._can_use_grouped_mm(selected,module.experts.gate_up_proj.transpose(-2,-1),offsets)
        dispatch=('torch.nn.functional.grouped_mm' if hasattr(F,'grouped_mm') else 'torch._grouped_mm') if native else 'transformers grouped_mm fallback'
        fused=moe._grouped_linear(selected,module.experts.gate_up_proj,offsets,is_transposed=module.experts.is_transposed)
        gate,up=fused.chunk(2,dim=-1)
        act=module.experts._apply_gate(fused)
        down=moe._grouped_linear(act,module.experts.down_proj,offsets,is_transposed=module.experts.is_transposed)
        terms=down*weights.reshape(-1)[perm,None]
        inv=torch.empty_like(perm)
        inv[perm]=torch.arange(len(perm),device=device)
        for name,values in (('expert_gate',gate),('expert_up',up),('expert_act',act),('expert_down',down),('combine_terms',terms)):
            ops[name]=values[inv].reshape(t,k,-1)
        out=ops['combine_terms'].sum(dim=1)
    ops['output']=out
    actual=module(x.unsqueeze(0)).squeeze(0)
    if not torch.equal(out,actual):
        raise RuntimeError(f'{implementation} mirror != actual Transformers module (max abs {(out.float()-actual.float()).abs().max().item()})')
    for name,shape in expected_shapes(c,t).items():
        if tuple(ops[name].shape)!=shape or not torch.isfinite(ops[name]).all():
            raise RuntimeError(f'invalid op: {name}')
    return ops,dispatch


def dump_backend(export: Path, destination: Path, dtype_name='float32', device='cpu', implementation='grouped_mm'):
    require_version()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        c,arrays=load_export(export)
        ops,dispatch=compute_backend(c,arrays,DTYPES[dtype_name],device,implementation)
        if device=='mps': torch.mps.synchronize()
        elapsed=time.perf_counter()-start
        entries=[]
        for name in OPS:
            integer=name=='topk_ids'
            save_array(destination/f'{name}.npy',ops[name],integer=integer)
            entries.append(dict(name=name,file=f'{name}.npy',shape=list(ops[name].shape),
                                logical_dtype='int32' if integer else ('fp32' if name=='router_probs' or dtype_name=='float32' else 'bf16'),
                                stage='selection' if OPS.index(name)<6 else 'post-selection'))
        manifest=dict(run_id=destination.name,backend=f'torch-{device}',
                      export=dict(path=str(export),config_sha256=sha256(export/'config.json')),
                      numerics=dict(dtype=dtype_name,experts_implementation=implementation,grouped_dispatch=dispatch,
                                    mirror_equals_module=True,cpu_fallback_enabled=False,
                                    torch_num_threads=torch.get_num_threads(),
                                    moe_source_sha256=sha256(Path(inspect.getsourcefile(moe))),
                                    olmoe_source_sha256=sha256(Path(inspect.getsourcefile(OlmoeSparseMoeBlock)))),
                      tokens=len(arrays['token_ids']),ops=entries,bug=None,
                      wall_seconds=elapsed,provenance=collect(dict(device=device,implementation=implementation)))
        write_json(destination/'manifest.json',manifest)
        write_json(destination/'status.json',dict(status='PASS',wall_seconds=elapsed))
        print(f'PASS {device}/{implementation}/{dtype_name}: mirror == module; {len(arrays["token_ids"])} tokens; {elapsed:.2f}s; {dispatch}',flush=True)
        return manifest
    except Exception as exc:
        write_json(destination/'status.json',dict(status='FAIL',device=device,implementation=implementation,
                                                  dtype=dtype_name,error=f'{type(exc).__name__}: {exc}',
                                                  wall_seconds=time.perf_counter()-start))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--dump',type=Path,required=True)
    p.add_argument('--dtype',choices=DTYPES,default='float32')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--implementation',choices=['eager','grouped_mm'],default='grouped_mm')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(a.threads)
    dump_backend(a.export,a.dump,a.dtype,a.device,a.implementation)

if __name__=='__main__':main()
