"""torchrun entrypoint: compare real NCCL sums with untouched port expectations."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def max_ulp(actual,expected,dtype):
    x=np.abs(expected.astype(np.float64))
    if dtype=='fp32':unit=np.abs(np.spacing(expected.astype(np.float32))).astype(np.float64)
    else:
        with np.errstate(divide='ignore'):exponent=np.floor(np.log2(x))
        unit=np.exp2(np.maximum(exponent,-126)-7)
    unit=np.maximum(unit,2.0**(-149 if dtype=='fp32' else -133))
    return float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))/unit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank);dist.init_process_group('nccl')
    results=[]
    for target in ('oproj','down'):
        for dtype,tdtype in [('fp32',torch.float32),('bf16',torch.bfloat16)]:
            prefix=f'{target}-n{world}-{dtype}'
            parts=np.load(a.input/f'{prefix}-partials.npy',mmap_mode='r')
            value=torch.tensor(np.array(parts[rank]),device=f'cuda:{rank}',dtype=tdtype)
            dist.all_reduce(value,op=dist.ReduceOp.SUM);torch.cuda.synchronize()
            actual=value.float().cpu().numpy()
            expected=np.load(a.input/f"{prefix}-{os.environ['NCCL_ALGO'].lower()}.npy")
            bitwise=bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32)))
            results.append(dict(target=target,n=world,rank=rank,dtype=dtype,algo=os.environ['NCCL_ALGO'],
                                bitwise_match=bitwise,mismatched_elements=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))),
                                elements=actual.size,max_ulp=max_ulp(actual,expected,dtype)))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f'rank{rank}.json').write_text(json.dumps(dict(results=results,gpu=torch.cuda.get_device_name(rank),
                  nccl_version=torch.cuda.nccl.version(),torch_version=torch.__version__,cuda_version=torch.version.cuda),indent=2))
    dist.destroy_process_group()
if __name__=='__main__':main()
