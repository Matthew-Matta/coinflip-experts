"""Offline routediff demo. Requires base+[dev] and moe_run; no Torch or weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime,timezone
from routediff.synthetic import make_export


def run(out: Path,binary: Path):
    binary=binary.resolve()
    if not binary.is_file():raise FileNotFoundError('Build port/build/moe_run with CMake before make demo')
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    export=make_export(out/'export',hidden=16,inter=8,experts=8,top_k=2,tokens=32,seed=42)
    reference=out/'reference';candidate=out/'candidate'
    base=out/'reference-numerics.json';base.write_text(json.dumps(dict(preset='hf_fp32')))
    altered=out/'candidate-numerics.json';altered.write_text(json.dumps(dict(preset='hf_fp32',gate_out='bf16')))
    for numerics,dump in ((base,reference),(altered,candidate)):
        subprocess.run([str(binary),'--export',str(export),'--numerics',str(numerics),'--dump',str(dump),'--threads','2'],check=True)
    rerun=shlex.join([str(binary),'--export',str(export),'--numerics',str(altered),'--threads','2','--dump','{out}','{patch}'])
    subprocess.run([sys.executable,'-m','routediff','compare',str(reference),str(candidate),
                    '--out',str(out/'report'),'--rerun',rerun],check=True)
    report=json.loads((out/'report/report.json').read_text())
    if report['localized_op']!='router_logits' or not report['patch_checks'][-1]['restored']:
        raise RuntimeError('demo failed to localise and repair the known precision mismatch')
    summary=dict(kind='Synthetic offline demo, not model evidence or blinded evaluation',
                 localized_op=report['localized_op'],restored=True,report=str(out/'report/report.md'),
                 binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),threads=2,seed=42)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"Localised op: {summary['localized_op']}")
    print(f"Report: {summary['report']}")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=Path('port/build/moe_run'))
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    out=a.out or Path('data/demo')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run(out,a.binary)
if __name__=='__main__':main()
