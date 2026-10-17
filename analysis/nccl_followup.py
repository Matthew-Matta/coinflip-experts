"""Summarise saved-rank NCCL outputs; order matches are descriptive, not identification."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.nccl_orders import explain,size_error,reduction
from reference.nccl_worker import max_ulp
from reference.format import sha256,write_json


def run(source: Path,partials: Path,out: Path):
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text())
    shares=[];positions=[];errors=[];analyses=[];rank_checks=[];positional=[]
    for n in (2,4):
        for algo in ('Ring','Tree'):
            folder=source/'nccl'/f'n{n}-{algo}'
            for target in ('oproj','down'):
                for wire in ('fp32','bf16'):
                    prefix=f'{target}-n{n}-{wire}'
                    parts_path=partials/f'{prefix}-partials.npy'
                    expected_path=partials/f'{prefix}-{algo.lower()}.npy'
                    for path in (parts_path,expected_path):
                        if sha256(path)!=summary['input_sha256']['nccl_input_sha256'][path.name]:raise ValueError('partial/expectation changed')
                    parts=np.load(parts_path);expected=np.load(expected_path)
                    # Analyse every distinct rank output; never assume all ranks agree.
                    groups={}
                    for rank in range(n):
                        path=folder/f'{prefix}-rank{rank}.npy'
                        digest=sha256(path)
                        if digest!=summary['output_sha256'][str(path.relative_to(source))]:raise ValueError('saved NCCL output changed')
                        groups.setdefault(digest,[]).append(rank)
                    rank_checks.append(dict(n=n,algo=algo,target=target,wire=wire,unique_rank_outputs=len(groups),rank_groups=list(groups.values())))
                    for digest,ranks in groups.items():
                        path=folder/f'{prefix}-rank{ranks[0]}.npy';actual=np.load(path)
                        tag=dict(n=n,algo=algo,target=target,wire=wire,ranks=ranks,actual_sha256=digest)
                        different=actual.view('u4')!=expected.view('u4')
                        errors.append(dict(**tag,elements=actual.size,mismatched_elements=int(different.sum()),
                                           bit_exact=not bool(different.any()),max_reference_ulp=max_ulp(actual,expected,wire),
                                           **size_error(actual,expected,parts,wire)))
                        if n!=4:continue
                        result=explain(parts,actual,expected,wire)
                        if algo=='Ring' and actual.size%8==0:
                            # Post-hoc pattern observed in the exhaustive bin analysis.
                            # This does not alter the frozen emulator or identify NCCL channels.
                            sequence=[(1,2,3,0),(2,3,0,1),(3,0,1,2),(0,1,2,3)]
                            flat=parts.reshape(4,-1);width=actual.size//8;prediction=np.empty(actual.size,'f4')
                            for block in range(8):
                                sl=slice(block*width,(block+1)*width)
                                prediction[sl]=reduction(flat[:,sl],sequence[block%4],'sequential',wire)
                            exact=np.array_equal(prediction.view('u4'),actual.reshape(-1).view('u4'))
                            positional.append(dict(**tag,elements=actual.size,blocks=8,elements_per_block=width,
                                                   orders=[list(sequence[b%4]) for b in range(8)],bit_exact=bool(exact)))
                        result['perfect_orders_by_position_bin']=[
                            [dict(order=c['order'],tree=c['tree']) for c in result['candidates'] if c['contiguous_bin_shares'][b]==1.]
                            for b in range(result['position_bins'])]
                        analyses.append({**tag,**result})
                        for c in result['candidates']:
                            row={**tag,'order':''.join(map(str,c['order'])),'tree':c['tree'],'share':c['share'],
                                 'matched_elements':c['matched_elements'],'bin_share_range':c['bin_share_range']}
                            if result['position_bins']%4==0:
                                rates=np.array(c['contiguous_bin_shares']).reshape(4,-1)
                                counts=np.array(result['position_bin_counts']).reshape(4,-1)
                                quarter=(rates*counts).sum(axis=1)/counts.sum(axis=1)
                                row.update({f'quarter{q}_share':float(rate) for q,rate in enumerate(quarter)})
                            shares.append(row)
                            for index,rate in enumerate(c['contiguous_bin_shares']):positions.append({k:row[k] for k in ('n','algo','target','wire','ranks','order','tree')} | dict(bin=index,bin_share=rate))
    write_json(out/'position-pattern.json',dict(label='Post-hoc descriptive positional fit, not a port change or verified NCCL channel mapping',checks=positional))
    write_json(out/'orders.json',dict(label='4xL4 on Modal, SHM transport',rank_checks=rank_checks,analyses=analyses,
               caveat='Equivalent summation orders overlap; matches do not identify NCCL implementation or channels.'))
    for name,rows in [('order_shares',shares),('position_shares',positions),('errors',errors)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    rows=[]
    for a in analyses:
        best=max(a['candidates'],key=lambda c:c['share'])
        rows.append(dict(algo=a['algo'],target=a['target'],wire=a['wire'],ranks=a['ranks'],
                         best_order=best['order'],best_tree=best['tree'],best_share=best['share'],
                         best_bin_min=min(best['contiguous_bin_shares']),best_bin_max=max(best['contiguous_bin_shares']),
                         any_order_share=a['any_order_share']))
    result=dict(label='4xL4 on Modal, SHM transport',order_summary=rows,errors=errors,analysis_source_sha256={p.name:sha256(p) for p in (Path(__file__),Path('analysis/nccl_orders.py'))},
                source_summary_sha256=sha256(source/'summary.json'),source_metadata={k:summary[k] for k in ('gpu','torch_version','cuda_version','nccl_version','wall_seconds','source_sha256')})
    from coinflip.provenance import save_run
    save_run(pd.DataFrame(errors),out,dict(source_summary_sha256=result['source_summary_sha256'],position_bins=64,candidates=48,threads=2),name='errors')
    write_json(out/'summary.json',result)
    print(json.dumps(rows,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','partials','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.partials,a.out)
