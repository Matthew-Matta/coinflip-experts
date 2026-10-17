"""Prompt-cluster intervals and accuracy/communication model for H4 emulation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.collective_cost import allreduce_cost
from coinflip.provenance import save_run
from reference.format import sha256, write_json


def bootstrap_flips(flipped, prompts, *, seed=20261016, resamples=10000):
    flipped=np.asarray(flipped,dtype=bool)
    prompts=np.asarray(prompts)
    if flipped.ndim!=1 or prompts.shape!=flipped.shape or not len(flipped):
        raise ValueError('one prompt label per nonempty token vector required')
    groups, index=np.unique(prompts,return_inverse=True)
    sizes=np.bincount(index)
    events=np.bincount(index,weights=flipped)
    draws=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=resamples)
    rates=(draws@events)/(draws@sizes)
    lo,hi=np.quantile(rates,[.025,.975])
    return dict(flips=int(flipped.sum()),tokens=len(flipped),prompts=len(groups),
                flip_rate=float(flipped.mean()),ci_low=float(lo),ci_high=float(hi))


def run(source: Path, out: Path, seed=20261016, resamples=10000):
    out.mkdir(parents=True,exist_ok=False)
    table=pd.read_parquet(source/'h4_collective_sweep.parquet')
    hashes={str(source/'h4_collective_sweep.parquet'):sha256(source/'h4_collective_sweep.parquet')}
    rows=[]
    for layer in sorted(table.layer.unique()):
        root=source/'runs'/f'layer{layer}'
        manifest=json.loads((root/'oracle/manifest.json').read_text())
        export=Path(manifest['export']['path'])
        cfg=json.loads((export/'config.json').read_text())
        meta=json.loads((export/'inputs/token_meta.json').read_text())
        prompts=[str(m['prompt_id']) for m in meta]
        oracle=np.sort(np.load(root/'oracle/topk_ids.npy'),axis=1)
        for p in (export/'config.json',export/'inputs/token_meta.json',root/'oracle/topk_ids.npy'):
            hashes[str(p)]=sha256(p)
        for record in table[table.layer==layer].to_dict('records'):
            n=int(record['n']);target=record['target']
            path=root/('n1' if n==1 else f"{target}-n{n}-{record['topology']}-{record['wire']}")/'topk_ids.npy'
            ids=np.sort(np.load(path),axis=1)
            if ids.shape!=oracle.shape or len(prompts)!=len(ids):raise ValueError('token shape mismatch')
            flipped=np.any(ids!=oracle,axis=1)
            stats=bootstrap_flips(flipped,prompts,seed=seed,resamples=resamples)
            assert stats['flips']==record['flips_vs_oracle'] and stats['tokens']==record['tokens']
            hashes[str(path)]=sha256(path)
            cost={} if n==1 else allreduce_cost(len(ids)*cfg['hidden_size']*(cfg['top_k'] if target=='down' else 1),
                                               n,record['wire'],record['topology'],1e-6,25e9)
            rows.append(dict(layer=int(layer),target=target,n=n,topology=record['topology'],wire=record['wire'],
                             **stats,aggregate_sent_bytes=cost.get('aggregate_sent_bytes',0),
                             critical_path_bytes=cost.get('critical_path_bytes',0),
                             sequential_steps=cost.get('sequential_steps',0),
                             predicted_seconds=cost.get('predicted_seconds',0)))
    frame=pd.DataFrame(rows)
    save_run(frame,out,dict(kind='collective emulation + alpha-beta model; not measured interconnect',
             bootstrap_unit='prompt',resamples=resamples,seed=seed,confidence=.95,
             alpha_seconds=1e-6,bandwidth_bytes_per_second=25e9,input_sha256=hashes),name='h4_intervals')
    frame.to_csv(out/'h4_intervals.csv',index=False)
    # Public provenance excludes local host and absolute paths.
    prov=out/'h4_intervals.provenance.json'
    value=json.loads(prov.read_text());value.pop('host',None);write_json(prov,value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(table.layer.unique()),figsize=(11,4.5),squeeze=False,sharey=True)
    for ax,layer in zip(axes[0],sorted(table.layer.unique()),strict=True):
        for wire,color in [('bf16','#b04a21'),('fp32','#2468a0')]:
            for topology,style,marker in [('ring','-','o'),('tree','--','s')]:
                x=frame[(frame.layer==layer)&(frame.target=='oproj')&(frame.wire==wire)&(frame.topology==topology)].sort_values('n')
                xx=x.aggregate_sent_bytes/1e6
                ax.plot(xx,100*x.flip_rate,style,marker=marker,color=color,label=f'{wire}, {topology}')
                ax.fill_between(xx,100*x.ci_low,100*x.ci_high,color=color,alpha=.1)
                if wire=='bf16' and topology=='ring':
                    for _,r in x.iterrows():ax.annotate(f"n={int(r.n)}",(r.aggregate_sent_bytes/1e6,100*r.flip_rate),fontsize=7,xytext=(2,5),textcoords='offset points')
        ax.set(xscale='log',xlabel='Modelled aggregate bytes sent (MB)',title=f'OLMoE layer {layer}')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set_ylabel('Route-set flips vs fp64 oracle (%)')
    fig.suptitle('Router-input collective: numerical emulation + communication model')
    fig.text(.5,.015,'Shading: prompt-cluster 95% CI, 10,000 resamples. No interconnect measurement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(out/'accuracy_vs_bytes.png',dpi=180);fig.savefig(out/'accuracy_vs_bytes.svg');plt.close(fig)
    svg=out/'accuracy_vs_bytes.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();frame=run(a.source,a.out);print(f'Wrote {len(frame)} configuration intervals and chart to {a.out}')
if __name__=='__main__':main()
