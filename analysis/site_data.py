"""Extract compact site data from frozen reports and saved dumps; never runs a model.

Run: python -m analysis.site_data [--out site/data]. Raw dumps are local-only.
Reported batch statistics are copied, never recalculated. Token metrics are display
extractions, with the same scaled-ULP definition as the frozen evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPS = ('o_proj_out', 'resid_sum', 'router_input', 'router_logits', 'router_probs',
       'topk_ids', 'topk_weights', 'expert_gate', 'expert_up', 'expert_act',
       'expert_down', 'combine_terms', 'output')
SLOTS = set(OPS[6:12])


class Sources:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.paths = {'analysis/site_data.py'}

    def path(self, path):
        p = Path(str(path).replace('<repo>/', ''))
        if p.is_absolute():
            # Frozen reports contain the original workstation prefix.
            parts = p.parts
            p = Path(*parts[parts.index('data'):])
        if not (self.root / p).is_dir():
            self.paths.add(p.as_posix())
        return self.root / p

    def read(self, path):
        return json.loads(self.path(path).read_text())

    def array(self, directory, op):
        return np.load(self.path(Path(directory) / (op + '.npy')), mmap_mode='r', allow_pickle=False)

    def finish(self, value):
        sources = []
        for p in sorted(self.paths):
            h = hashlib.sha256()
            with (self.root / p).open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            sources.append(dict(path=p, sha256=h.hexdigest()))
        return dict(schema_version=1, **value, provenance=dict(sources=sources))


def scrub(value):
    """Remove historical absolute repo prefixes, preserving numeric report fields."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '').replace('<repo>/', '')
    return value


def pipeline():
    s = Sources()
    manifest = s.read('data/dumps/layer0-torch-float32/manifest.json')
    s.path('port/src/block.cpp')
    s.path('reference/dump_block.py')
    names = ['Attention projection', 'Residual addition', 'Normalize', 'Router scores',
             'Expert probabilities', 'Choose eight experts', 'Routing weights', 'Expert gate',
             'Expert up projection', 'Expert activation', 'Expert down projection',
             'Weight expert outputs', 'Combine experts']
    explanations = ['Project the captured attention output back to the model width.',
                    'Add the incoming residual to the attention projection.',
                    'Scale each token using RMSNorm before routing.',
                    'Score the token against all 64 experts.',
                    'Turn the scores into probabilities with softmax.',
                    'Select the eight largest expert probabilities.',
                    'Normalize the selected probabilities to sum to one.',
                    'Compute the gating half of each selected expert.',
                    'Compute the value half of each selected expert.',
                    'Apply SiLU to the gate and multiply by the up projection.',
                    'Project each expert back to the model width.',
                    'Multiply each expert output by its routing weight.',
                    'Add the weighted experts in ascending expert ID order.']
    lines = [297, 409, 58, 352, 353, 354, 356, 332, 332, 333, 334, 335, 336]
    ops = [dict(id=op, display_name=names[i], explanation=explanations[i],
                shape=manifest['ops'][i]['shape'][1:], dtype=manifest['ops'][i]['logical_dtype'],
                pytorch=f'transformers 5.12.1 modeling_olmoe.py:{lines[i]}',
                cpp='port/src/block.cpp::run_block') for i, op in enumerate(OPS)]
    edges = [(OPS[i], OPS[i+1]) for i in range(6)]
    edges += [('router_input', 'expert_gate'), ('router_input', 'expert_up'),
              ('topk_ids', 'expert_gate'), ('topk_ids', 'expert_up'),
              ('expert_gate', 'expert_act'), ('expert_up', 'expert_act'),
              ('expert_act', 'expert_down'), ('expert_down', 'combine_terms'),
              ('topk_weights', 'combine_terms'), ('combine_terms', 'output')]
    return s.finish(dict(ops=ops, edges=edges))


def token_state(s, ref, cand, index, report):
    rid = s.array(ref, 'topk_ids')[index].astype(int)
    cid = s.array(cand, 'topk_ids')[index].astype(int)
    same = set(rid) == set(cid)
    manifest = s.read(Path(ref) / 'manifest.json')
    types = {x['name']: x['logical_dtype'] for x in manifest['ops']}
    ops = {}
    for op in OPS:
        tol = report['ops'][op].get('tolerance_scaled_ulp')
        if op == 'topk_ids' or (op in SLOTS and not same):
            flagged = False
            if op == 'topk_ids' and not same:
                probs = s.array(cand, 'router_probs')[index]
                mask = np.zeros(len(probs), dtype=bool)
                mask[rid] = True
                flagged = bool(probs[mask].min() >= probs[~mask].max())
            ops[op] = dict(max_scaled_ulp=None, tolerance_scaled_ulp=tol,
                           compared=op == 'topk_ids', flagged=flagged)
            continue
        a = s.array(ref, op)[index].astype(np.float64)
        b = s.array(cand, op)[index].astype(np.float64)
        if op in SLOTS:
            b = b[[list(cid).index(e) for e in rid]]
        magnitude = np.maximum(np.abs(a), np.sqrt(np.mean(a*a)))
        mant, emin = {'fp32': (23, -126), 'bf16': (7, -126), 'fp16': (10, -14), 'fp64': (52, -1022)}[types[op]]
        _, exponent = np.frexp(np.where(magnitude > 0, magnitude, 2.**emin))
        spacing = np.ldexp(1., np.maximum(exponent-1, emin)-mant)
        error = float(np.max(np.abs(a-b)/spacing))
        ops[op] = dict(max_scaled_ulp=error, tolerance_scaled_ulp=tol,
                       compared=True, flagged=tol is not None and error > tol)
    return dict(batch=scrub(report), source_runs=dict(reference=str(ref), candidate=str(cand)),
                token=dict(ops=ops, route_sets_equal=same),
                routes=dict(reference=sorted(rid.tolist()), candidate=sorted(cid.tolist())))


def scenarios():
    s = Sources()
    ref = 'data/dumps/layer0-torch-float32'
    cand = 'data/results/simd/layer0-cpp-hf_fp32-scalar'
    p = s.array(ref, 'router_probs').astype(np.float64)
    same = (np.sort(s.array(ref, 'topk_ids'), axis=1) == np.sort(s.array(cand, 'topk_ids'), axis=1)).all(axis=1)
    ordered = np.sort(p, axis=1)[:, ::-1]
    margins = (ordered[:, 7]-ordered[:, 8])/ordered[:, 7]
    index = int(np.argmin(np.where(same & (margins > 0), margins, np.inf)))
    metadata = s.read('data/export/olmoe-1b-7b-0924/layer0/inputs/token_meta.json')[index]
    token = dict(layer=0, index=index, prompt_id=metadata['prompt_id'], position=metadata['position'],
                 split=metadata['split'], normalized_margin=float(margins[index]),
                 probabilities=p[index].tolist(), top8=sorted(s.array(ref, 'topk_ids')[index].astype(int).tolist()))
    scores = s.read('data/evaluation/real-v2/scores.json')
    benign = next(x for x in scores if x['layer'] == 0 and x['kind'] == 'benign')
    clean_path = f"data/evaluation/real-v2/cases/{benign['case_id']}/report/report.json"
    clean = s.read(clean_path)
    sealed = s.read('data/evaluation/real-v2/sealed_reports.json')
    if hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest() != sealed[benign['case_id']]:
        raise ValueError('sealed clean report changed')
    # Frozen benign report is the batch view; the near-tie token has its own full-dump view.
    items = [dict(id='clean', label='Clean C++ vs PyTorch fp32', bug_type=None,
                  source_report=clean_path, token_index=index,
                  before=token_state(s, ref, cand, index, clean), after=None)]
    items[0]['batch_scope'] = 'Separate sealed benign evaluation batch; token view uses the full layer-0 scalar C++ dump.'
    bug = next(x for x in scores if x['layer'] == 0 and x.get('bug_type') == 'router_accum')
    path = f"data/evaluation/real-v2/cases/{bug['case_id']}/report/report.json"
    report = s.read(path)
    if hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest() != sealed[bug['case_id']]:
        raise ValueError('sealed planted-bug report changed')
    br, bc = s.path(report['reference_run']).relative_to(ROOT), s.path(report['candidate_run']).relative_to(ROOT)
    mismatches = (np.sort(s.array(br, 'topk_ids'), axis=1) != np.sort(s.array(bc, 'topk_ids'), axis=1)).any(axis=1)
    bi = int(np.flatnonzero(mismatches)[0])
    # Temporary patched dumps were deleted by the original evaluator. Do not invent values.
    proof = report['patch_checks'][-1]
    after = dict(batch=dict(patch_check=proof), source_runs=None, token=None, routes=None,
                 unavailable='Original evaluator deleted temporary patched dumps; only the sealed restoration check survives.')
    items.append(dict(id='planted_bug', label='Sealed real-v2 router fault', bug_type='router_accum',
                      source_report=path, batch_scope='Sealed real-v2 evaluation prompt', token_index=bi, before=token_state(s, br, bc, bi, report), after=after))
    path = 'findings/torch-patch-forward-bf16/hook-demo-mps-router/report.json'
    report = s.read(path)
    base = 'data/results/torch-hook-demo-mps-router-postlayout'
    before = token_state(s, base+'/clean', base+'/faulty', 0, report)
    after = token_state(s, base+'/clean', base+'/repaired', 0, report)
    # Batch after metrics weren't saved; copy the existing bit-exact restoration summary.
    after['batch'] = s.read('findings/torch-patch-forward-bf16/hook-demo-mps-router/summary.json')
    items.append(dict(id='apple_gpu_fault', label='Apple GPU forward-hook fault (known demo)',
                      bug_type='router_logits sign inversion', source_report=path, batch_scope='Known 64-token forward-hook demo', token_index=0,
                      before=before, after=after))
    ex = s.read('findings/backend-pairs-v1/bf16_ties.json')['0']['examples'][0]
    t = ex['token']; tied = ex['tied_group']
    mac = next(x for x in s.read('findings/topk-portability/macos.json')['rows'] if x['layer'] == 0 and x['token'] == t)
    linux = next(x for x in s.read('findings/topk-portability/linux.json')['rows'] if x['layer'] == 0 and x['token'] == t)
    sets = dict(cpp=sorted(s.array('data/dumps/layer0-cpp-hf_bf16', 'topk_ids')[t].astype(int).tolist()),
                apple_gpu=sorted(s.array('data/dumps/real-backend-pairs-v1/layer0-mps-eager-bfloat16', 'topk_ids')[t].astype(int).tolist()),
                pytorch_macos=mac['stored_probs_chosen_set'], pytorch_linux=linux['stored_probs_chosen_set'])
    tie = dict(layer=0, index=t, probabilities=s.array('data/dumps/layer0-torch-bfloat16', 'router_probs')[t, tied].astype(float).tolist(),
               tied_experts=tied, choices={k: sorted(set(v) & set(tied)) for k, v in sets.items()}, route_sets=sets,
               caveat='Descriptive platform/build comparison; tie-breaking is unspecified. Apple probabilities are not bit-identical to CPU probabilities.')
    return s.finish(dict(token=token, scenarios=items, exact_tie=tie))


def census():
    s = Sources(); a = s.read('findings/census-olmoe-prefill/summary.json')
    b = s.read('findings/census-olmoe-prefill/structure.json')
    return s.finish(dict(phase=a['phase'], results=a['results'], structure=b['results']))


def accumulator():
    s = Sources(); rows = []
    with s.path('findings/accumulator-width/per_op_bits.csv').open() as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k == 'mantissa_bits': row[k] = 7 if v == 'kahan e8m7' else int(v)
                elif k in ('flips_layer0', 'flips_layer7', 'tokens_per_layer'): row[k] = int(v)
                elif k in ('router_logit_rel_l2_max', 'output_rel_l2_max'): row[k] = float(v)
                elif k.startswith('calibrated_flagged') or k == 'patch_checked': row[k] = v == 'True'
            row['accumulator'] = 'o_proj' if row['accumulator'] == 'oproj' else row['accumulator']
            row['flips_total'] = row['flips_layer0'] + row['flips_layer7']
            rows.append(row)
    return s.finish(dict(rows=rows))


def nccl():
    s = Sources(); a = s.read('findings/nccl-validation/order-followup/summary.json')
    b = s.read('findings/nccl-validation/order-followup/position-pattern.json')
    tree = next(x for x in a['order_summary'] if x['algo'] == 'Tree')
    return s.finish(dict(ranks=tree['ranks'], tree_order=tree['best_order'], ring_blocks=b['checks'],
                         order_summary=a['order_summary'], n2=[x for x in a['errors'] if x['n'] == 2], caveat=b['label']))


HEADLINES = [
    ('census', 'Batch size changes expert choices', 'findings/census-olmoe-prefill/summary.json', ['results.bs8.flip_rate', 'results.r0b.flip_rate']),
    ('debugger', 'Planted faults detected and localized', 'findings/routediff-eval-real-v2/summary.json', ['detected', 'localized', 'effective_bug_runs', 'false_alarms', 'benign_runs']),
    ('accumulator', 'Router accumulator precision', 'findings/accumulator-width/summary.json', ['router.min_bits_for_zero_flips', 'router.calibrated_detects_at_m', 'router.flips_vs_oracle_by_m.7', 'router.tokens']),
    ('nccl', 'Recovered four-rank summation order', 'findings/nccl-validation/order-followup/summary.json', ['order_summary']),
    ('qwen', 'Second model port', 'findings/qwen3-port/h_port_fp32.json', ['tokens', 'route_mismatch_tokens_all'])]


def field(obj, path):
    for key in path.split('.'): obj = obj[key]
    return obj


def headline():
    s = Sources(); items = []
    for id_, label, path, fields in HEADLINES:
        obj = s.read(path)
        items.append(dict(id=id_, label=label, values={k: field(obj, k) for k in fields}, source=path, source_fields=fields))
    return s.finish(dict(items=items))


def generate(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in ('pipeline', 'scenarios', 'census', 'accumulator', 'nccl', 'headline'):
        value = globals()[name]()
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
        total += len(raw)
        (out / (name+'.json')).write_bytes(raw)
    if total >= 300000: raise ValueError(f'payload {total} exceeds budget')
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'site/data')
    print(f'{generate(parser.parse_args().out)} bytes')
