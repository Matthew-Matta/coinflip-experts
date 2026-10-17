"""Descriptive layer and expert-swap structure from the frozen prefill arrays."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from reference.format import sha256,write_json


def structure(source: Path,count=200):
    results={}
    inputs={}
    for label in ('bs8','bs32'):
        layer_counts=np.zeros(16,dtype=np.int64);swap_hist=np.zeros(9,dtype=np.int64)
        layer_hist=np.zeros(17,dtype=np.int64);tokens=0
        for i in range(count):
            ref=source/'r0a'/f'{i:03d}.npy';cand=source/label/f'{i:03d}.npy'
            a=np.load(ref);b=np.load(cand)
            if a.shape!=b.shape or a.shape[1:]!=(16,8):raise ValueError('unpaired route arrays')
            # Each route has 8 distinct IDs, so this counts removed (= added) experts.
            swaps=8-(a[:,:,:,None]==b[:,:,None,:]).any(axis=-1).sum(axis=-1)
            different=swaps>0;layer_counts+=different.sum(axis=0)
            swap_hist+=np.bincount(swaps.reshape(-1),minlength=9)
            layer_hist+=np.bincount(different.sum(axis=1),minlength=17)
            tokens+=len(a)
            inputs[str(ref)]=sha256(ref);inputs[str(cand)]=sha256(cand)
        n=int(layer_counts.sum())
        results[label]=dict(tokens=tokens,token_layers=tokens*16,differing_sets=n,
             per_layer_flips=layer_counts.tolist(),per_layer_rates=(layer_counts/tokens).tolist(),
             replaced_experts_histogram=swap_hist.tolist(),one_expert_swaps=int(swap_hist[1]),
             one_expert_share=float(swap_hist[1]/n),flipped_layers_per_token_histogram=layer_hist.tolist(),
             tokens_flipped_any_layer=int(tokens-layer_hist[0]),token_flip_fraction=float(1-layer_hist[0]/tokens),
             tokens_flipped_all_layers=int(layer_hist[16]))
    return dict(kind='descriptive structure; fixed captured corpus, not op-level attribution',
                unit='token-layer top-k set; swap size = experts removed (= experts added)',results=results,input_sha256=inputs)


def write_per_layer(result, destination: Path):
    """Write the published layer table from the same structure counts."""
    with destination.open('w', newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['layer','tokens','bs8_flips','bs8_rate','bs32_flips','bs32_rate'])
        a,b=(result['results'][label] for label in ('bs8','bs32'))
        for layer in range(16):
            writer.writerow([layer,a['tokens'],a['per_layer_flips'][layer],a['per_layer_rates'][layer],
                             b['per_layer_flips'][layer],b['per_layer_rates'][layer]])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=structure(a.source);a.out.mkdir(parents=True,exist_ok=True)
    write_json(a.out/'structure.json',result)
    write_per_layer(result,a.out/'per_layer.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.8,4.4))
    for label,color in [('bs8','#b34f25'),('bs32','#2468a0')]:
        ax.plot(range(16),100*np.array(result['results'][label]['per_layer_rates']),marker='o',label=label,color=color)
    ax.set(xlabel='MoE layer',ylabel='Captured route-set flips vs bs1 (%)',xticks=range(16),
           title='OLMoE prefill: descriptive layer structure')
    ax.legend();ax.grid(alpha=.2)
    fig.text(.5,.015,'200 fixed prompts · 66,207 tokens/layer · SGLang 0.5.21 + backport · one L4',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for suffix in ('png','svg'):fig.savefig(a.out/f'per_layer_flips.{suffix}',dpi=180)
    svg=a.out/'per_layer_flips.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps({k:{x:v for x,v in r.items() if x!='input_sha256'} for k,r in result['results'].items()},indent=2))
if __name__=='__main__':main()
