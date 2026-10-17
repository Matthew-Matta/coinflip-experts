"""Freeze real L0 partials and verify every expectation against committed moe_run."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from evaluation.run import subset_export,file_hash
from reference.format import write_json


def prepare(source: Path,build_root: Path,out: Path):
    if subprocess.check_output(['git','-C',str(build_root),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('build worktree has uncommitted tracked sources')
    head=subprocess.check_output(['git','-C',str(build_root),'rev-parse','HEAD'],text=True).strip()
    out.mkdir(parents=True,exist_ok=False);inp=out/'input';inp.mkdir();parts=out/'partials';parts.mkdir()
    meta=json.loads((source/'inputs/token_meta.json').read_text());seen={};indices=[]
    for i,m in enumerate(meta):
        p=m['prompt_id'];n=seen.get(p,0)
        if n<2:indices.append(i);seen[p]=n+1
    export=subset_export(source,inp/'export',np.array(indices))
    write_json(inp/'selection.json',dict(indices=indices,prompts=len(seen),tokens=len(indices),method='First two valid positions per original prompt'))
    num=inp/'numerics.json';write_json(num,dict(preset='hf_fp32'))
    binary=build_root/'port/build/moe_run'
    helper=out/'collective_partials'
    subprocess.run(['clang++','-std=c++17','-O2','-I'+str(build_root/'port/include'),
                    'reference/collective_partials.cpp',str(build_root/'port/build/libcoinflip_port.a'),'-o',str(helper)],check=True)
    subprocess.run([str(binary),'--export',str(export),'--numerics',str(num),'--dump',str(inp/'clean'),'--threads','4'],check=True)
    subprocess.run([str(helper),str(export),str(inp/'clean'),str(parts)],check=True)
    checks=[]
    for n in (2,4):
        for wire in ('fp32','bf16'):
            for target in ('oproj','down'):
                for topo in ('ring','tree'):
                    name=f'{target}-n{n}-{wire}-{topo}';cfg=out/(name+'.json');dump=out/'moe-validation'/name
                    write_json(cfg,dict(preset='hf_fp32',collective_n=n,collective_topology=topo,collective_wire=wire,collective_on=target))
                    subprocess.run([str(binary),'--export',str(export),'--numerics',str(cfg),'--dump',str(dump),'--threads','4'],check=True)
                    expected=np.load(parts/(name+'.npy'));actual=np.load(dump/('o_proj_out.npy' if target=='oproj' else 'expert_down.npy')).reshape(expected.shape)
                    if not np.array_equal(actual.view('u4'),expected.view('u4')):raise ValueError(f'helper != moe_run: {name}')
                    checks.append(dict(config=name,partial_helper_equals_moe_run=True))
    write_json(out/'moe_validation.json',checks)
    write_json(out/'preparation.json',dict(head=head,binary_sha256=file_hash(binary),helper_source_sha256=file_hash(Path('reference/collective_partials.cpp')),
               input_sha256={p.name:file_hash(p) for p in parts.glob('*.npy')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/export/olmoe-1b-7b-0924/layer0'))
    p.add_argument('--build-root',type=Path,default=Path('../coinflip-eval-build'))
    p.add_argument('--out',type=Path,default=Path('data/nccl'))
    a=p.parse_args();prepare(a.source,a.build_root,a.out)
if __name__=='__main__':main()
