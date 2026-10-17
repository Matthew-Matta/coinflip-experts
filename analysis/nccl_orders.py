"""CPU-only, descriptive order explanations for saved four-rank NCCL tensors."""
import argparse
from itertools import permutations
import json
from pathlib import Path
import numpy as np


def rounded(x,wire):
    x=np.asarray(x,dtype=np.float32)
    if wire=='fp32':return x
    if wire!='bf16':raise ValueError('wire must be fp32 or bf16')
    bits=np.ascontiguousarray(x).view(np.uint32).astype(np.uint64)
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return bits.astype(np.uint32).view(np.float32)


def reduction(parts,order,tree,wire):
    add=lambda a,b:rounded(np.add(a,b,dtype=np.float32),wire)
    a,b,c,d=[parts[i] for i in order]
    if tree=='pairwise':return add(add(a,b),add(c,d))
    acc=rounded(np.zeros_like(a)+a,wire)
    return add(add(add(acc,b),c),d)


def size_error(actual,emulation,parts,wire):
    eps=2.**(-23 if wire=='fp32' else -7)
    scale=np.abs(parts.astype(np.float64)).sum(axis=0)
    error=np.abs(actual.astype(np.float64)-emulation.astype(np.float64))
    if np.any((scale==0)&(error!=0)):raise ValueError('nonzero result error for an all-zero partial tuple')
    relative=np.divide(error,scale,out=np.zeros_like(error),where=scale!=0)
    # eps units: requested magnitude normalisation, not the ULP of a cancelled output.
    units=relative/eps
    return dict(max_relative_to_sum_abs=float(relative.max()),max_size_scaled_wire_ulp=float(units.max()),
                p99_size_scaled_wire_ulp=float(np.quantile(units,.99)),
                definition='abs(nccl-emulation) / sum(abs(partials)) / eps_wire; eps_fp32=2^-23, eps_bf16=2^-7')


def explain(parts,actual,emulation,wire,bins=64):
    parts=np.asarray(parts,dtype=np.float32)
    actual=np.asarray(actual,dtype=np.float32);emulation=np.asarray(emulation,dtype=np.float32)
    if parts.shape[0]!=4 or actual.shape!=parts.shape[1:] or emulation.shape!=actual.shape:
        raise ValueError('need 4 paired partials, NCCL outputs, and emulation tensors')
    if not all(np.isfinite(v).all() for v in (parts,actual,emulation)):raise ValueError('nonfinite input')
    n=actual.size
    if n==0:raise ValueError("empty tensor")
    bins=min(bins,n)
    position=np.arange(n,dtype=np.int64)*bins//n
    counts=np.bincount(position,minlength=bins);any_match=np.zeros(n,dtype=bool);multiplicity=np.zeros(n,dtype=np.uint8)
    candidates=[]
    for order in permutations(range(4)):
        for tree in ('sequential','pairwise'):
            predicted=reduction(parts,order,tree,wire)
            match=(predicted.view(np.uint32)==actual.view(np.uint32)).reshape(-1)
            any_match|=match;multiplicity+=match
            by_bin=np.bincount(position,weights=match,minlength=bins)
            rates=np.divide(by_bin,counts,out=np.zeros(bins),where=counts!=0)
            candidates.append(dict(order=list(order),tree=tree,matched_elements=int(match.sum()),share=float(match.mean()),
                                   contiguous_bin_shares=rates.tolist(),bin_share_range=float(rates.max()-rates.min())))
    return dict(elements=n,wire=wire,candidate_count=48,candidates=candidates,
                any_order_explains=int(any_match.sum()),any_order_share=float(any_match.mean()),
                matching_order_multiplicity=np.bincount(multiplicity,minlength=49).tolist(),
                position_bins=bins,position_bin_counts=counts.tolist(),position_definition='Equal contiguous bins of C-order flattened tensor; not known NCCL channels',
                size_aware_error=size_error(actual,emulation,parts,wire),
                caveat='Multiple orders can produce the same rounded value. Match shares do not identify the executed NCCL order or channel mapping.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('partials','actual','emulation','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wire',choices=('fp32','bf16'),required=True)
    a=p.parse_args();result=explain(np.load(a.partials),np.load(a.actual),np.load(a.emulation),a.wire)
    from reference.format import sha256
    result['input_sha256']={str(x):sha256(x) for x in (a.partials,a.actual,a.emulation)}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
