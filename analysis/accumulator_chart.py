"""Plot only a completed accumulator sweep; never promote partial rows.jsonl."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reference.format import sha256,write_json

LABELS={'oproj':'o_proj','rms':'RMSNorm sum','router':'Router','expert':'Experts (post-selection)','combine':'Combine (post-selection)'}


def load_complete(source: Path):
    parquet=source/'accumulator_width.parquet';summary=source/'summary.json';prov=source/'accumulator_width.provenance.json'
    if not all(p.is_file() for p in (parquet,summary,prov)):
        raise FileNotFoundError('Sweep is not complete: require final parquet, summary and provenance. Partial rows.jsonl is not plotted.')
    config=json.loads(prov.read_text())['config'];frame=pd.read_parquet(parquet)
    expected={(int(layer),knob,f'e8m{m}') for layer in config['layers'] for knob in config['knobs'] for m in config['mantissas']}
    selected=frame[frame.variant.str.match(r'^e8m\d+$')].copy()
    if selected.duplicated(['layer','knob','variant','split']).any():raise ValueError('duplicate sweep observations')
    for split in ('all','evaluation'):
        found=set(map(tuple,selected[selected.split==split][['layer','knob','variant']].itertuples(index=False,name=None)))
        if found!=expected:raise ValueError(f'incomplete {split} sweep grid')
    selected=selected[selected.split=='all'].copy();selected['mantissa_bits']=selected.variant.str[3:].astype(int)
    if (selected.tokens<=0).any() or (selected.flips_vs_oracle>selected.tokens).any():raise ValueError('invalid counts')
    selected['flip_rate']=selected.flips_vs_oracle/selected.tokens
    return selected,config


def run(source: Path,out: Path):
    frame,config=load_complete(source)
    out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layers=sorted(frame.layer.unique())
    fig,axes=plt.subplots(1,len(layers),figsize=(6*len(layers),4.6),squeeze=False,sharey=True)
    for ax,layer in zip(axes[0],layers,strict=True):
        for knob,label in LABELS.items():
            data=frame[(frame.layer==layer)&(frame.knob==knob)].sort_values('mantissa_bits')
            if len(data):ax.plot(data.mantissa_bits,100*data.flip_rate,marker='o',ms=3,label=label)
        ax.set(xlabel='Accumulator fraction bits (8-bit exponent)',title=f'Layer {layer}',xticks=list(range(7,24,2)),
               yscale='symlog',ylim=(-.002,None));ax.set_yscale('symlog',linthresh=.02)
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0,0].set_ylabel('Route-set flips vs fp64 oracle (%)')
    fig.suptitle('Emulated accumulator precision, one operation group narrowed at a time (exploratory)')
    fig.text(.5,.015,'7 bits = bf16 precision; 23 = fp32. Post-selection faults cannot change this block’s routes. Layers not pooled.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    for suffix in ('png','svg'):fig.savefig(out/f'flips_vs_bits.{suffix}',dpi=180)
    svg=out/'flips_vs_bits.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n');plt.close(fig)
    frame[['layer','knob','mantissa_bits','tokens','flips_vs_oracle','flip_rate']].to_csv(out/'flips_vs_bits.csv',index=False)
    write_json(out/'metadata.json',dict(kind='Exploratory emulation, descriptive counts; no interconnect measurement',
               split='all',layers=config['layers'],base=config['base'],mantissas=config['mantissas'],
               source_sha256={p.name:sha256(p) for p in source.glob('accumulator_width.*') if p.suffix in ('.json','.parquet')}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.source,a.out)
if __name__=='__main__':main()
