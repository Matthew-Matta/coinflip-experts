"""Run real CPU-grouped and MPS-eager dumps, recording unsupported paths explicitly."""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import torch
from reference.backend_pairs import dump_backend
from reference.format import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    a.out.mkdir(parents=True,exist_ok=False)
    results=[]
    for export in a.exports:
        for dtype in ('float32','bfloat16'):
            for device,impl in (('cpu','grouped_mm'),('mps','eager')):
                dest=a.out/f'{export.name}-{device}-{impl}-{dtype}'
                try:
                    manifest=dump_backend(export,dest,dtype,device,impl)
                    results.append(dict(run=dest.name,status='PASS',tokens=manifest['tokens']))
                except Exception as exc:
                    print(f'FAIL {dest.name}: {type(exc).__name__}: {exc}',flush=True)
                    results.append(dict(run=dest.name,status='FAIL',error=f'{type(exc).__name__}: {exc}'))
                gc.collect()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
    write_json(a.out/'summary.json',results)
    if any(r['status']!='PASS' for r in results): raise SystemExit(1)

if __name__=='__main__':main()
