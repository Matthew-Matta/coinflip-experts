"""Bounded-memory descriptive comparisons of the real backend dumps; no causal claims."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from coinflip.provenance import collect
from reference.format import write_json, sha256
from routediff.dumps import SLOT_ALIGNED, align_slots, same_route_sets


def summarize_pair(reference: Path, candidate: Path) -> dict:
    r=json.loads((reference/'manifest.json').read_text())
    c=json.loads((candidate/'manifest.json').read_text())
    if r['tokens']!=c['tokens'] or r['export']['config_sha256']!=c['export']['config_sha256']:
        raise ValueError('pair does not share export configuration and token count')
    rid=np.load(reference/'topk_ids.npy',mmap_mode='r')
    cid=np.load(candidate/'topk_ids.npy',mmap_mode='r')
    same=same_route_sets(rid,cid)
    logits=np.load(reference/'router_logits.npy',mmap_mode='r')
    k=rid.shape[1]
    sorted_logits=np.sort(logits,axis=-1)
    ties=sorted_logits[:,-k]==sorted_logits[:,-k-1]
    ops={}
    for op in r['ops']:
        name=op['name']
        if name=='topk_ids': continue
        a=np.load(reference/op['file'],mmap_mode='r')
        b=np.load(candidate/op['file'],mmap_mode='r')
        if a.shape!=b.shape: raise ValueError(f'{name}: different shapes')
        num=den=max_abs=0.0
        compared=0
        for start in range(0,len(a),32):
            end=min(start+32,len(a))
            aa=np.asarray(a[start:end],dtype=np.float64)
            bb=np.asarray(b[start:end],dtype=np.float64)
            valid=np.ones(end-start,dtype=bool)
            if name in SLOT_ALIGNED:
                valid=same[start:end]
                bb=align_slots(rid[start:end],cid[start:end],bb,valid)
            aa,bb=aa[valid],bb[valid]
            compared+=len(aa)
            if not len(aa): continue
            diff=aa-bb
            max_abs=max(max_abs,float(np.max(np.abs(diff))))
            num+=float(np.sum(diff*diff));den+=float(np.sum(aa*aa))
        ops[name]=dict(max_abs=max_abs,relative_l2=float(np.sqrt(num/den)) if den else None,
                       compared_tokens=compared,exact=max_abs==0)
    return dict(reference=reference.name,candidate=candidate.name,tokens=len(same),
                backend=c['backend'],dtype=c['numerics']['dtype'],
                experts_implementation=c['numerics']['experts_implementation'],
                route_mismatch_tokens=int((~same).sum()),reference_cutoff_tie_tokens=int(ties.sum()),
                ops=ops,reference_manifest_sha256=sha256(reference/'manifest.json'),
                candidate_manifest_sha256=sha256(candidate/'manifest.json'),
                interpretation='Descriptive differences; no tolerance calibration or patch-forward causal claim. Slot-aligned ops exclude tokens whose route sets differ.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--cpu-root',type=Path,default=Path('data/dumps'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for manifest in sorted(a.pairs.glob('*/manifest.json')):
        c=json.loads(manifest.read_text())
        config=json.loads((Path(c['export']['path'])/'config.json').read_text())
        ref=a.cpu_root/f'layer{config["layer"]}-torch-{c["numerics"]["dtype"]}'
        result=summarize_pair(ref,manifest.parent)
        rows.append(result)
        print(result['candidate'],'routes',result['route_mismatch_tokens'],'/',result['tokens'],
              'output relative L2',result['ops']['output']['relative_l2'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists(): raise FileExistsError(a.out)
    write_json(a.out,dict(pairs=rows,provenance=collect({'analysis':'real backend pairs'})))

if __name__=='__main__':main()
