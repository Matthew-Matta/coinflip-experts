"""Calibrate, freeze, run blinded cases, seal reports, then score.

python -m evaluation.run --exports data/export/olmoe-1b-7b-0924/layer0 data/export/olmoe-1b-7b-0924/layer7 \
 --references data/dumps/layer0-torch-float32 data/dumps/layer7-torch-float32 --out data/evaluation/run1
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import shutil
import os
import subprocess
import sys

import numpy as np
from coinflip.provenance import collect
from reference.format import OPS, save_array, sha256, write_json
from routediff.compare import compare
from routediff.dumps import load_run
from routediff.localize import calibrate
from evaluation.broker import replay_broker
from evaluation.scoring import score_case, summarize

BENIGN = [dict(preset='hf_fp32'),
          dict(preset='hf_fp32', oproj_order='pairwise', router_order='pairwise', expert_order='pairwise'),
          dict(preset='hf_fp32', oproj_fma=False, router_fma=False, expert_fma=False)]


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integrity(export: Path, reference: Path) -> dict:
    files = [export/'config.json', export/'inputs/token_meta.json', reference/'manifest.json']
    files += sorted((export/'weights').glob('*.npy')) + sorted((export/'inputs').glob('*.npy'))
    files += sorted(reference.glob('*.npy'))
    return {str(path): file_hash(path) for path in files}


def choose_cases(registry: list[dict], prompts: list[str], seed: int, repeats: int, layer: int, experts: int = 64) -> list[dict]:
    rng = random.Random(seed)
    result = []
    for _ in range(repeats):
        shuffled = list(registry)
        rng.shuffle(shuffled)
        for bug in shuffled:
            params = {}
            if 'dtype' in bug['description']:
                params['dtype'] = 'bf16'
            if bug['type'] == 'expert_permuted':
                a,b = rng.sample(range(experts),2)
                params = dict(a=a, b=b)
            if bug['type'] == 'combine_order':
                params = dict(order='expert_desc')
            result.append(dict(case_id=f'{rng.getrandbits(128):032x}', kind='bug',
                               prompt_id=rng.choice(prompts), layer=layer,
                               bug_type=bug['type'], injected_op=bug['injected_op'], params=params,
                               seed=rng.getrandbits(32)))
    return result


def subset_export(source: Path, out: Path, indices: np.ndarray) -> Path:
    out.mkdir(parents=True)
    (out / 'weights').symlink_to((source / 'weights').resolve(), target_is_directory=True)
    (out / 'inputs').mkdir()
    for name in ('o_proj_in', 'residual', 'token_ids'):
        arr = np.load(source / 'inputs' / f'{name}.npy', mmap_mode='r', allow_pickle=False)[indices]
        save_array(out / 'inputs' / f'{name}.npy', arr, integer=name == 'token_ids')
    meta = json.loads((source / 'inputs/token_meta.json').read_text())
    write_json(out / 'inputs/token_meta.json', [meta[i] for i in indices])
    cfg = json.loads((source / 'config.json').read_text())
    cfg['subset'] = dict(source_config_sha256=sha256(source / 'config.json'), indices=indices.tolist())
    write_json(out / 'config.json', cfg)
    return out


def subset_reference(source: Path, out: Path, export: Path, indices: np.ndarray) -> Path:
    out.mkdir(parents=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    for op in manifest['ops']:
        arr = np.load(source / op['file'], mmap_mode='r', allow_pickle=False)[indices]
        save_array(out / op['file'], arr, integer=op['name'] == 'topk_ids')
        op['shape'] = list(arr.shape)
    manifest.update(run_id=out.name, tokens=len(indices),
                    export=dict(path=str(export), config_sha256=sha256(export / 'config.json')))
    manifest['source_manifest_sha256'] = sha256(source / 'manifest.json')
    manifest['source_token_indices'] = indices.tolist()
    write_json(out / 'manifest.json', manifest)
    return out


def execute(argv: list[str], log: Path, timeout=600, **kwargs):
    with log.open('w') as stream:
        subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True, **kwargs)


def run(exports: list[Path], references: list[Path], out: Path, *, seed=20261016, repeats=1,
        binary=Path('port/build/moe_run'), resamples=10000, margin=1.0):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be finite and positive')
    if len(exports) != len(references) or not exports or repeats < 1:
        raise ValueError('need corresponding exports/references and positive repeats')
    exports, references = [p.resolve() for p in exports], [p.resolve() for p in references]
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    private = out.parent / (out.name + '-private')
    private.mkdir(mode=0o700)
    (private / 'logs').mkdir()
    original_binary = binary.resolve()
    binary = private / 'moe_run'
    shutil.copy2(original_binary, binary)
    project = Path(__file__).resolve().parents[1]
    runtime = out / 'runtime'
    runtime.mkdir()
    shutil.copytree(project/'routediff', runtime/'routediff', ignore=shutil.ignore_patterns('__pycache__'))
    localiser_hashes = {str(p.relative_to(runtime)): sha256(p) for p in sorted(runtime.rglob('*.py'))}
    registry = json.loads(subprocess.check_output([str(binary), '--list-bugs'], text=True))
    write_json(private / 'registry.json', registry)
    settings = dict(seed=seed, repeats=repeats, margin=margin, baseline='torch-cpu float32 vs C++ hf_fp32',
                    calibration_configs=BENIGN, binary_sha256=sha256(binary),
                    localiser_source_sha256=localiser_hashes,
                    evaluator_source_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                    input_integrity=[integrity(e,r) for e,r in zip(exports,references,strict=True)],
                    references=[dict(path=str(p), sha256=sha256(p/'manifest.json')) for p in references],
                    inputs_previously_inspected_in_M2=True,
                    interpretation='Held-out prompt split for calibration/evaluation; not previously unseen inputs.',
                    blinding='Localiser receives only opaque replay URLs; logical separation, not an OS security sandbox.',
                    provenance=collect({'evaluation_seed': seed}))
    write_json(out / 'settings.json', settings)
    base_num = out / 'baseline.json'
    write_json(base_num, BENIGN[0])
    calibration_pairs, contexts, truths = [], {}, []
    for source, ref_source in zip(exports, references, strict=True):
        cfg = json.loads((source / 'config.json').read_text())
        layer = cfg['layer']
        meta = json.loads((source / 'inputs/token_meta.json').read_text())
        cal = np.array([i for i,m in enumerate(meta) if m['split'] == 'calibration'])
        evaluation_prompts = sorted({str(m['prompt_id']) for m in meta if m['split'] == 'evaluation'})
        if not len(cal) or not evaluation_prompts:
            raise ValueError('both prompt splits must be nonempty')
        cal_prompts = {str(meta[i]['prompt_id']) for i in cal}
        if cal_prompts.intersection(evaluation_prompts):
            raise ValueError('a prompt occurs in both splits')
        layer_dir = out / f'layer{layer}'
        layer_dir.mkdir()
        ce = subset_export(source, layer_dir/'cal-export', cal)
        cr = subset_reference(ref_source, layer_dir/'cal-reference', ce, cal)
        for n, config in enumerate(BENIGN):
            num = layer_dir / f'cal-{n}.json'
            write_json(num, config)
            cand = layer_dir / f'cal-candidate-{n}'
            execute([str(binary), '--export', str(ce), '--numerics', str(num), '--dump', str(cand), '--threads', '4'],
                    private/'logs'/f'cal-{layer}-{n}.log')
            calibration_pairs.append((load_run(cr), load_run(cand)))
        cases = choose_cases(registry, evaluation_prompts, seed + layer, repeats, layer, cfg['num_experts'])
        # One independent benign pair per held-out prompt, cycling allowed variants.
        rng = random.Random(seed + layer + 1000)
        for j, prompt in enumerate(evaluation_prompts):
            cases.append(dict(case_id=f'{rng.getrandbits(128):032x}', kind='benign', prompt_id=prompt,
                              layer=layer, config=BENIGN[j % len(BENIGN)]))
        for truth in cases:
            prompt = truth['prompt_id']
            key = (layer, prompt)
            if key not in contexts:
                root = layer_dir / f'prompt-{len(contexts):04d}'
                root.mkdir()
                indices = np.array([i for i,m in enumerate(meta) if str(m['prompt_id']) == prompt])
                pe = subset_export(source, root/'export', indices)
                pr = subset_reference(ref_source, root/'reference', pe, indices)
                clean = root/'clean'
                execute([str(binary), '--export', str(pe), '--numerics', str(base_num), '--dump', str(clean), '--threads', '4'],
                        private/'logs'/f'clean-{len(contexts)}.log')
                contexts[key] = dict(export=pe, reference=pr, clean=clean)
            truth['context'] = key
        truths.extend(cases)
    tolerances = calibrate(calibration_pairs, margin=margin)
    del calibration_pairs
    write_json(out/'tolerances.json', tolerances)
    tolerance_hash = sha256(out/'tolerances.json')
    write_json(out/'frozen.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                                     settings_sha256=sha256(out/'settings.json'), tolerances_sha256=tolerance_hash))
    (out/'tolerances.json').chmod(0o444)
    write_json(private/'truth.json', truths)
    cases_dir = out/'cases'
    cases_dir.mkdir()
    commands, reports = {}, {}
    for truth in truths:
        ticket = truth['case_id']
        ctx = contexts[tuple(truth['context'])]
        cmd = [str(binary.resolve()), '--export', str(ctx['export'].resolve()), '--threads', '4']
        if truth['kind'] == 'bug':
            spec = private / f'{ticket}.json'
            write_json(spec, dict(bug_id=ticket, type=truth['bug_type'], params=truth['params'],
                                  injected_op=truth['injected_op'], seed=truth['seed']))
            cmd += ['--numerics', str(base_num.resolve()), '--bug', str(spec.resolve())]
        else:
            num = private / f'{ticket}.json'
            write_json(num, truth['config'])
            cmd += ['--numerics', str(num.resolve())]
        commands[ticket] = cmd
    with replay_broker(commands, private/'logs') as endpoint:
        for n, truth in enumerate(truths):
            ticket = truth['case_id']
            ctx = contexts[tuple(truth['context'])]
            root = cases_dir/ticket
            root.mkdir()
            try:
                execute(commands[ticket]+['--dump', str(root/'candidate')], private/'logs'/f'{ticket}-initial.log')
                replay = shlex.join([sys.executable, '-m', 'evaluation.replay', '--endpoint', endpoint+'/'+ticket,
                                     '--dump', '{out}', '{patch}'])
                execute([sys.executable, '-m', 'routediff', 'compare', str(ctx['reference']), str(root/'candidate'),
                         '--out', str(root/'report'), '--tolerances', str(out/'tolerances.json'), '--rerun', replay],
                        root/'localiser.log', timeout=1200, cwd=runtime,
                        env={**os.environ, 'PYTHONPATH': str(project)})
                reports[ticket] = json.loads((root/'report/report.json').read_text())
            except Exception as exc:
                reports[ticket] = dict(error=type(exc).__name__)
            print(f'Completed blinded case {n+1}/{len(truths)}: {ticket}', flush=True)
    # All localiser runs finish before truth is read for scoring/effect classification.
    write_json(out/'sealed_reports.json', {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                                          for k,v in reports.items()})
    if sha256(out/'tolerances.json') != tolerance_hash:
        raise RuntimeError('frozen tolerances changed during evaluation')
    rows = []
    for truth in json.loads((private/'truth.json').read_text()):
        ticket = truth['case_id']
        report = reports[ticket]
        if report.get('error'):
            rows.append(dict(case_id=ticket, prompt_id=truth['prompt_id'], layer=truth['layer'], kind=truth['kind'],
                             bug_type=truth.get('bug_type'), effect=None, detected=False, correct_localization=False,
                             error=report['error']))
            continue
        effect = None
        if truth['kind'] == 'bug':
            ctx = contexts[tuple(truth['context'])]
            # Independent ground truth: bug vs its clean backend, not the localiser verdict vs torch.
            effect = bool(compare(load_run(ctx['clean']), load_run(cases_dir/ticket/'candidate'), tolerances)['flagged_ops'])
        rows.append(score_case(report, truth, effect))
    result = summarize(rows, seed, resamples)
    result['frozen_tolerances_sha256'] = tolerance_hash
    result['settings_sha256'] = sha256(out/'settings.json')
    write_json(out/'scores.json', rows)
    write_json(out/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('by_bug_type','cluster_bootstrap')}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exports', nargs='+', type=Path, required=True)
    p.add_argument('--references', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--margin', type=float, default=1.0,
                   help='Positive multiplier for benign calibration tolerances (default: 1.0).')
    p.add_argument('--binary', type=Path, default=Path('port/build/moe_run'))
    a = p.parse_args()
    run(a.exports, a.references, a.out, seed=a.seed, repeats=a.repeats, binary=a.binary, margin=a.margin)

if __name__ == '__main__':
    main()
