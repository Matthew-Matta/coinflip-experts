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
import numpy as np

import torch
import torch.nn.functional as F
from transformers.integrations import moe
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from coinflip.provenance import collect
from reference.dump_block import DTYPES, make_modules, require_version
from reference.format import OPS, expected_shapes, load_export, save_array, sha256, write_json


@torch.inference_mode()
def compute_backend(c, arrays, dtype, device, implementation, patches=None, patch_ids=None, demo_scale_combine=None, _legacy_expert_layout=False):
    from reference.patch_hooks import PatchHooks
    if device == 'mps':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('MPS CPU fallback is enabled; refusing to mislabel fallback execution')
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in this process')
    module, norm = make_modules(c, arrays, dtype, implementation=implementation, device=device)
    tap=PatchHooks(patches,patch_ids,demo_scale_combine)
    tensor=lambda name: torch.from_numpy(arrays[name]).to(device=device,dtype=dtype)
    ops={}
    def record(name,value,ids=None):
        value=tap(name,value,ids);ops[name]=value;return value
    oo=record('o_proj_out',F.linear(tensor('o_proj_in'),tensor('o_proj')))
    residual=record('resid_sum',tensor('residual')+oo)
    x=record('router_input',norm(residual))
    logits=record('router_logits',F.linear(x,module.gate.weight))
    probs=record('router_probs',F.softmax(logits,dim=-1,dtype=torch.float32))
    weights,ids=torch.topk(probs,c['top_k'],dim=-1)
    ids=record('topk_ids',ids.to(torch.int32)).long()
    # A patched selection must gather the probabilities for its new expert IDs.
    weights=torch.gather(probs,1,ids)
    if c['norm_topk_prob']:weights=weights/weights.sum(dim=-1,keepdim=True)
    weights=record('topk_weights',weights.to(dtype),ids)
    t,k,i,h=len(x),c['top_k'],c['intermediate_size'],c['hidden_size']
    dispatch='eager'
    if implementation=='eager':
        mask=F.one_hot(ids,num_classes=c['num_experts']).permute(2,1,0)
        hits=[];gate=torch.empty((t,k,i),dtype=dtype,device=device);up=torch.empty_like(gate)
        for e in torch.greater(mask.sum(dim=(-1,-2)),0).nonzero():
            e=e[0];slots,tokens=torch.where(mask[e]);hits.append((e,slots,tokens))
            g,u=F.linear(x[tokens],module.experts.gate_up_proj[e]).chunk(2,dim=-1)
            gate[tokens,slots]=g;up[tokens,slots]=u
        gate=record('expert_gate',gate,ids);up=record('expert_up',up,ids)
        # Elementwise math is applied in each original expert's row layout.
        act=torch.empty_like(gate)
        for e,slots,tokens in hits:
            g,u=gate[tokens,slots],up[tokens,slots]
            if not _legacy_expert_layout:
                # Preserve the chunk(2) strides of Transformers' fused projection.
                # A contiguous gate copy can select a different SIMD SiLU kernel.
                g,u=torch.cat((g,u),dim=-1).chunk(2,dim=-1)
            act[tokens,slots]=module.experts.act_fn(g)*u
        act=record('expert_act',act,ids)
        down=torch.empty((t,k,h),dtype=dtype,device=device)
        for e,slots,tokens in hits:down[tokens,slots]=F.linear(act[tokens,slots],module.experts.down_proj[e])
        down=record('expert_down',down,ids)
        terms=record('combine_terms',down*weights[:,:,None],ids)
        out=torch.zeros_like(x)
        for e,slots,tokens in hits:out.index_add_(0,tokens,terms[tokens,slots].to(out.dtype))
    else:
        expert_ids,perm=torch.sort(ids.reshape(-1));selected=x[perm//k]
        histc_input=expert_ids.float() if device in ('cpu','mps') else expert_ids.int()
        counts=torch.histc(histc_input,bins=c['num_experts'],min=0,max=c['num_experts']-1)
        offsets=torch.cumsum(counts,dim=0,dtype=torch.int32)
        native=moe._can_use_grouped_mm(selected,module.experts.gate_up_proj.transpose(-2,-1),offsets)
        dispatch=('torch.nn.functional.grouped_mm' if hasattr(F,'grouped_mm') else 'torch._grouped_mm') if native else 'transformers grouped_mm fallback'
        inv=torch.empty_like(perm);inv[perm]=torch.arange(len(perm),device=device)
        def slot(values):return values[inv].reshape(t,k,-1)
        def grouped(values):return values.reshape(t*k,-1)[perm]
        fused=moe._grouped_linear(selected,module.experts.gate_up_proj,offsets,is_transposed=module.experts.is_transposed)
        gate,up=fused.chunk(2,dim=-1)
        gate=record('expert_gate',slot(gate),ids);up=record('expert_up',slot(up),ids)
        act=module.experts._apply_gate(torch.cat([grouped(gate),grouped(up)],dim=-1))
        act=record('expert_act',slot(act),ids)
        down=moe._grouped_linear(grouped(act),module.experts.down_proj,offsets,is_transposed=module.experts.is_transposed)
        down=record('expert_down',slot(down),ids)
        terms=record('combine_terms',slot(grouped(down)*weights.reshape(-1)[perm,None]),ids)
        out=terms.sum(dim=1)
    record('output',out)
    if not patches and demo_scale_combine is None:
        if implementation=='eager':
            from reference.mirror_diagnostics import check_eager
            check_eager(module,x,ops)
        else:
            actual=module(x.unsqueeze(0)).squeeze(0)
            if not torch.equal(ops['output'],actual):
                raise RuntimeError(f'{implementation} mirror != actual Transformers module')
    for name,shape in expected_shapes(c,t).items():
        if tuple(ops[name].shape)!=shape or not torch.isfinite(ops[name]).all():raise RuntimeError(f'invalid op: {name}')
    return ops,dispatch


def dump_backend(export: Path, destination: Path, dtype_name='float32', device='cpu', implementation='grouped_mm', patches=None, patch_ids=None, demo_scale_combine=None):
    require_version()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    try:
        c,arrays=load_export(export)
        ops,dispatch=compute_backend(c,arrays,DTYPES[dtype_name],device,implementation,patches,patch_ids,demo_scale_combine)
        if device=='mps': torch.mps.synchronize()
        elapsed=time.perf_counter()-start
        entries=[]
        for name in OPS:
            integer=name=='topk_ids'
            save_array(destination/f'{name}.npy',ops[name],integer=integer)
            entries.append(dict(name=name,file=f'{name}.npy',shape=list(ops[name].shape),
                                logical_dtype='int32' if integer else ('fp32' if name=='router_probs' or dtype_name=='float32' else 'bf16'),
                                stage='selection' if OPS.index(name)<6 else 'post-selection'))
        skipped = 0
        if patch_ids is not None:
            source_ids = np.load(patch_ids, allow_pickle=False)
            current_ids = ops['topk_ids'].detach().cpu().numpy()
            mismatched = int(np.any(np.sort(source_ids,axis=-1)!=np.sort(current_ids,axis=-1),axis=-1).sum())
            from reference.patch_hooks import SLOT_OPS
            skipped = mismatched * len(SLOT_OPS.intersection(patches or {}))
        manifest=dict(run_id=destination.name,backend=f'torch-{device}',
                      export=dict(path=str(export),config_sha256=sha256(export/'config.json')),
                      numerics=dict(dtype=dtype_name,experts_implementation=implementation,grouped_dispatch=dispatch,
                                    mirror_equals_module=not bool(patches) and demo_scale_combine is None,cpu_fallback_enabled=False,
                                    demo_fault=None if demo_scale_combine is None else dict(op='combine_terms',scale=demo_scale_combine,blinded=False),
                                    instrumentation='forward-hook taps on the validated numerical mirror',
                                    patches={name:dict(path=str(path),sha256=sha256(Path(path))) for name,path in (patches or {}).items()},
                                    torch_num_threads=torch.get_num_threads(),
                                    moe_source_sha256=sha256(Path(inspect.getsourcefile(moe))),
                                    olmoe_source_sha256=sha256(Path(inspect.getsourcefile(OlmoeSparseMoeBlock)))),
                      tokens=len(arrays['token_ids']),ops=entries,bug=None,patch_skipped_token_ops=skipped,
                      wall_seconds=elapsed,provenance=collect(dict(device=device,implementation=implementation)))
        write_json(destination/'manifest.json',manifest)
        write_json(destination/'status.json',dict(status='PASS',wall_seconds=elapsed))
        validation = 'hooked mirror (patched/demo)' if patches or demo_scale_combine is not None else 'mirror == module'
        print(f'PASS {device}/{implementation}/{dtype_name}: {validation}; {len(arrays["token_ids"])} tokens; {elapsed:.2f}s; {dispatch}',flush=True)
        return manifest
    except Exception as exc:
        write_json(destination/'status.json',dict(status='FAIL',device=device,implementation=implementation,
                                                  dtype=dtype_name,error=f'{type(exc).__name__}: {exc}',
                                                  wall_seconds=time.perf_counter()-start))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--dump','--out',dest='dump',type=Path,required=True)
    p.add_argument('--patch',action='append',default=[])
    p.add_argument('--patch-ids',type=Path)
    p.add_argument('--demo-scale-combine',type=float,help='Known forward-hook fault for an unblinded demo only.')
    p.add_argument('--dtype',choices=DTYPES,default='float32')
    p.add_argument('--device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--implementation',choices=['eager','grouped_mm'],default='grouped_mm')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    patches=dict(item.split('=',1) for item in a.patch)
    dump_backend(a.export,a.dump,a.dtype,a.device,a.implementation,patches,a.patch_ids,a.demo_scale_combine)

if __name__=='__main__':main()
