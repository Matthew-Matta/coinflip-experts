"""Score sealed localiser reports; injection labels are read only after inference."""
from __future__ import annotations
from collections import Counter
import numpy as np


def counts(rows: list[dict]) -> dict:
    bugs = [r for r in rows if r['kind'] == 'bug' and r['effect'] is True]
    benign = [r for r in rows if r['kind'] == 'benign']
    detected = [r for r in bugs if r['detected']]
    n = dict(total_runs=len(rows), effective_bug_runs=len(bugs),
             detected=len(detected), localized=sum(r['correct_localization'] for r in detected),
             benign_runs=len(benign), false_alarms=sum(r['detected'] for r in benign),
             no_effect_runs=sum(r['kind'] == 'bug' and r['effect'] is False for r in rows),
             no_effect_alarms=sum(r['kind'] == 'bug' and r['effect'] is False and r['detected'] for r in rows),
             errors=sum(bool(r.get('error')) for r in rows))
    def rate(a, b):
        return n[a] / n[b] if n[b] else None
    n.update(detection_rate=rate('detected', 'effective_bug_runs'),
             localization_rate=rate('localized', 'detected'), false_positive_rate=rate('false_alarms', 'benign_runs'))
    return n


def summarize(rows: list[dict], seed: int, resamples: int = 10000) -> dict:
    result = counts(rows)
    groups = sorted({r['prompt_id'] for r in rows})
    rng = np.random.default_rng(seed)
    columns = ('effective_bug_runs', 'detected', 'localized', 'benign_runs', 'false_alarms')
    totals = np.array([[counts([r for r in rows if r['prompt_id'] == g])[k] for k in columns] for g in groups])
    samples = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=resamples) @ totals
    intervals = {}
    for name, a, b in (('detection_rate', 1, 0), ('localization_rate', 2, 1), ('false_positive_rate', 4, 3)):
        valid = samples[:, b] > 0
        values = samples[valid, a] / samples[valid, b]
        intervals[name] = list(np.quantile(values, [0.025, 0.975])) if len(values) else None
    result['cluster_bootstrap'] = dict(unit='prompt_id (shared across layers)', prompts=len(groups),
                                      seed=seed, resamples=resamples, intervals=intervals,
                                      limitation='Percentile intervals describe resampling of tested prompts; all-zero/all-one samples can be degenerate.')
    result['by_bug_type'] = {name: counts([r for r in rows if r.get('bug_type') == name])
                             for name in sorted({r['bug_type'] for r in rows if r['kind'] == 'bug'})}
    rates = [result[k] for k in ('detection_rate', 'localization_rate', 'false_positive_rate')]
    result['thresholds'] = dict(detection_min=0.9, localization_min=0.8, false_positive_max=0.05)
    result['passes_thresholds'] = bool(all(x is not None for x in rates) and not result['errors']
                                      and rates[0] >= .9 and rates[1] >= .8 and rates[2] <= .05)
    return result


def score_case(report: dict, truth: dict, effect: bool | None) -> dict:
    detected = bool(report.get('flagged_ops'))
    return dict(case_id=truth['case_id'], prompt_id=truth['prompt_id'], layer=truth['layer'],
                kind=truth['kind'], bug_type=truth.get('bug_type'), effect=effect,
                detected=detected, localized_op=report.get('localized_op'),
                correct_localization=bool(detected and report.get('localized_op') == truth.get('injected_op')),
                error=None)
