"""Known-hook-fault demo, not blinded evaluation. CPU by default; no auto MPS."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np
import torch
from evaluation.run import subset_export
from reference.backend_pairs import dump_backend
from reference.format import OPS,sha256,write_json


def run(source: Path,out: Path,tolerances: Path,device='cpu',fault='combine'):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((source/'inputs/token_meta.json').read_text());seen={};indices=[]
    for i,row in enumerate(meta):
        p=row['prompt_id']
        if seen.get(p,0)<2:indices.append(i);seen[p]=seen.get(p,0)+1
    export=subset_export(source,out/'export',np.array(indices))
    op='router_logits' if fault=='router' else 'combine_terms'
    scale=-1. if fault=='router' else 16.
    kwargs={'demo_scale_router':scale} if fault=='router' else {'demo_scale_combine':scale}
    clean=out/'clean';faulty=out/'faulty'
    dump_backend(export,clean,'bfloat16',device,'eager')
    dump_backend(export,faulty,'bfloat16',device,'eager',**kwargs)
    command=[sys.executable,'-m','reference.backend_pairs','--export',str(export),'--dtype','bfloat16',
             '--device',device,'--implementation','eager','--threads','2','--demo-scale-'+fault,str(scale),
             '--out','{out}','{patch}']
    subprocess.run([sys.executable,'-m','routediff','compare',str(clean),str(faulty),
                    '--out',str(out/'report'),'--tolerances',str(tolerances),'--rerun',shlex.join(command)],check=True)
    report=json.loads((out/'report/report.json').read_text())
    if report['localized_op']!=op or not report['patch_checks'][-1]['restored']:
        raise RuntimeError('demo did not localise/restore known hooked op')
    repaired=out/'repaired'
    dump_backend(export,repaired,'bfloat16',device,'eager',{op:clean/f'{op}.npy'},clean/'topk_ids.npy',**kwargs)
    exact=all(np.array_equal(np.load(clean/f'{name}.npy'),np.load(repaired/f'{name}.npy')) for name in OPS)
    if not exact:raise RuntimeError('patch did not restore clean computation bit-exactly')
    result=dict(kind='Known forward-hook demo, NOT blinded evaluation',device=device,threads=2,
                injected_op=op,scale=scale,localized_op=report['localized_op'],
                patch_restores_all_ops_bit_exact=True,route_sets_unchanged=report['route_mismatch_tokens']==0,
                route_mismatch_tokens=report['route_mismatch_tokens'],tokens=len(indices),prompts=len(seen),tolerance_sha256=sha256(tolerances),
                report=str(out/'report/report.md'))
    write_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,default=Path('data/export/olmoe-1b-7b-0924/layer0'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tolerances',type=Path,default=Path('findings/bf16-tolerances/tolerances-bf16-exploratory.json'))
    p.add_argument('--fault',choices=('combine','router'),default='combine')
    p.add_argument('--device',choices=('cpu','mps'),default='cpu')
    a=p.parse_args();run(a.export,a.out,a.tolerances,a.device,a.fault)
if __name__=='__main__':main()
