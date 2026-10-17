"""Descriptive legacy/fused-layout comparison; strict production checks stay on."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

import torch

from reference.backend_pairs import compute_backend
from reference.format import load_export, write_json
from reference.mirror_diagnostics import MirrorMismatch


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    # Exact deterministic fixture used by the three reported failing tests.
    from reference.tests.test_dump_block import tiny_export
    fixture_root=out/'fixture'; fixture_root.mkdir()
    export=tiny_export.__wrapped__(fixture_root)
    c,arrays=load_export(export)
    result=dict(torch=torch.__version__,platform=platform.platform(),
                torch_config=torch.__config__.show(),cpu_capability=torch.backends.cpu.get_cpu_capability(),
                lscpu=subprocess.run(['lscpu'],capture_output=True,text=True).stdout if platform.system()=='Linux' else None,
                cases=[])
    for threads in (1,2):
        torch.set_num_threads(threads)
        for legacy in (True,False):
            label=f'{"legacy-contiguous" if legacy else "fused-strided"}-threads{threads}'
            os.environ['COINFLIP_MIRROR_DIAGNOSTICS']=str(out/label)
            row=dict(layout=label,threads=threads)
            try:
                compute_backend(c,arrays,torch.float32,'cpu','eager',_legacy_expert_layout=legacy)
                row['bit_exact']=True
            except MirrorMismatch as exc:
                row.update(bit_exact=False,error=str(exc))
            result['cases'].append(row)
    os.environ.pop('COINFLIP_MIRROR_DIAGNOSTICS',None)
    write_json(out/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)
    # Legacy mismatches are descriptive. A corrected-path mismatch remains red.
    if any(not row['bit_exact'] for row in result['cases'] if row['layout'].startswith('fused')):
        raise RuntimeError('corrected fused-layout mirror still differs; see saved per-op arrays')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
if __name__=='__main__':main()
