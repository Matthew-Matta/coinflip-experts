"""Descriptive CPU top-k portability probe; a different chosen set is a result."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import torch
from reference.format import sha256, write_json


def probe(fixture: Path) -> dict:
    data=json.loads(fixture.read_text())
    if len(data['rows'])!=38:raise ValueError('expected all 38 tie rows')
    rows=[]
    for row in data['rows']:
        logits=torch.tensor(row['logits'],dtype=torch.float32)
        mac_probs=torch.tensor(row['macos_probs'],dtype=torch.float32)
        if logits.shape!=(64,) or not torch.isfinite(logits).all():raise ValueError('invalid logits')
        sorted_probs=torch.sort(mac_probs,descending=True).values
        if sorted_probs[7]!=sorted_probs[8]:raise ValueError('fixture row lacks boundary tie')
        # Direct probabilities isolate top-k from platform-dependent softmax math.
        direct=torch.topk(mac_probs,data['top_k']).indices.tolist()
        probs=torch.softmax(logits,dim=-1,dtype=torch.float32)
        recomputed=torch.topk(probs,data['top_k']).indices.tolist()
        expected=row['macos_chosen_set']
        rows.append(dict(layer=row['layer'],token=row['token'],macos_chosen_set=expected,
                         stored_probs_order=direct,stored_probs_chosen_set=sorted(direct),
                         stored_probs_same_set=sorted(direct)==expected,
                         logits_softmax_order=recomputed,logits_softmax_chosen_set=sorted(recomputed),
                         logits_softmax_same_set=sorted(recomputed)==expected,
                         softmax_bit_exact_to_macos=bool(torch.equal(probs,mac_probs))))
    return dict(kind='descriptive CPU top-k platform comparison; mismatch is not a test failure',
                platform=platform.system(),machine=platform.machine(),torch_version=torch.__version__,
                fixture_sha256=sha256(fixture),rows=rows,total_rows=len(rows),
                stored_probs_same_sets=sum(r['stored_probs_same_set'] for r in rows),
                logits_softmax_same_sets=sum(r['logits_softmax_same_set'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,default=Path('reference/tests/fixtures/bf16_topk_ties.json'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=probe(a.fixture);write_json(a.out,result)
    for row in result['rows']:print(json.dumps(row,sort_keys=True))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
