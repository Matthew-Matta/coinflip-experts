"""Freeze 64 paired real positions per layer for the bounded CUDA follow-up."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from evaluation.run import subset_export
from reference.backend_pairs import dump_backend
import torch
from reference.format import sha256, write_json


def prepare(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    selections={}
    for layer in (0,7):
        source=Path(f'data/export/olmoe-1b-7b-0924/layer{layer}')
        meta=json.loads((source/'inputs/token_meta.json').read_text()); seen={};indices=[]
        for i,row in enumerate(meta):
            p=row['prompt_id']
            if seen.get(p,0)<2:indices.append(i);seen[p]=seen.get(p,0)+1
        indices=np.array(indices)
        export=subset_export(source,out/f'layer{layer}/export',indices)
        for dtype in ('float32','bfloat16'):
            dump_backend(export,out/f'layer{layer}/cpu-{dtype}',dtype,'cpu','eager')
        selections[str(layer)]=dict(indices=indices.tolist(),tokens=len(indices),prompts=len(seen),method='First two valid positions per original prompt')
    shutil.copyfile('data/evaluation/real-v2/tolerances.json',out/'tolerances-float32.json')
    shutil.copyfile('findings/bf16-tolerances/tolerances-bf16-exploratory.json',out/'tolerances-bfloat16.json')
    hashes={}
    for p in sorted(out.rglob('*')):
        if p.is_file():hashes[str(p.relative_to(out))]=sha256(p)
    # pathlib rglob deliberately does not recurse into the frozen weight symlinks.
    for layer in (0,7):
        for p in sorted((out/f'layer{layer}/export/weights').glob('*.npy')):
            hashes[str(p.relative_to(out))]=sha256(p)
    result=dict(selection=selections,input_sha256=hashes,nccl_input_sha256={p.name:sha256(p) for p in sorted(Path('data/nccl/partials').glob('*.npy'))})
    write_json(out/'prepared.json',result)
    print(json.dumps(dict(files=len(hashes),bytes=sum((out/p).stat().st_size for p in hashes),selection=selections),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out)
