"""Paired prefill flip rates and excess over R0, clustered over prompts."""
import argparse
import json
from pathlib import Path
import numpy as np
from reference.format import write_json,sha256


def summarize(records,seed=20261017,resamples=10000):
    labels=('r0b','bs8','bs32');prompts=sorted({r['prompt_id'] for r in records})
    if len(prompts)!=200:raise ValueError('require all 200 frozen prompts')
    results={};draws=np.random.default_rng(seed).multinomial(len(prompts),np.full(len(prompts),1/len(prompts)),size=resamples)
    for label in labels:
        selected=[r for r in records if r['config']==label]
        rows={r['prompt_id']:r for r in selected}
        if set(rows)!=set(prompts) or len(selected)!=len(rows):raise ValueError('missing or duplicate prompt/config')
        units=np.array([rows[p]['units'] for p in prompts]);flips=np.array([rows[p]['flips'] for p in prompts]);r0=np.array([rows[p]['r0_flips'] for p in prompts])
        denom=draws@units;rates=(draws@flips)/denom;excess=(draws@(flips-r0))/denom
        results[label]=dict(flips=int(flips.sum()),units=int(units.sum()),flip_rate=float(flips.sum()/units.sum()),
                ci=list(np.quantile(rates,[.025,.975])),r0_flips=int(r0.sum()),
                excess_over_r0=float((flips-r0).sum()/units.sum()),excess_ci=list(np.quantile(excess,[.025,.975])),
                sequences_with_flip=sum(bool(rows[p]['any_flip']) for p in prompts),sequences=len(prompts),
                sequence_flip_fraction_ci=list(np.quantile((draws@np.array([bool(rows[p]['any_flip']) for p in prompts]))/len(prompts),[.025,.975])))
    ci=results['bs32']['excess_ci']
    return dict(phase='prefill only, not decode',model='OLMoE-1B-7B-0924',results=results,
                h1='PASS' if ci[0]>0 else 'FALSIFIED (95% CI includes zero)' if ci[0]<=0<=ci[1] else 'NOT SUPPORTED (negative excess)',
                robustness_upper_below_0_01_percent=bool(ci[1]<.0001),
                limitation='Zero-event percentile bootstrap can be degenerate; do not infer a general population upper bound. Results apply only to the frozen prompts and configuration.',
                seed=seed,resamples=resamples,bootstrap_unit='prompt',margins='not captured')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();source=json.loads(a.summary.read_text())
    if source['status']!='PASS' or source['mode']!='census':raise ValueError('census did not complete')
    result=summarize(source['counts']);result['source_sha256']=sha256(a.summary)
    write_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
