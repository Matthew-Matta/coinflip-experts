# Blinded planted-bug evaluation

Build `port/build/moe_run`, produce the reference exports/dumps, then run:

```sh
python -m evaluation.run \
  --exports data/export/olmoe-1b-7b-0924/layer0 data/export/olmoe-1b-7b-0924/layer7 \
  --references data/dumps/layer0-torch-float32 data/dumps/layer7-torch-float32 \
  --out data/evaluation/real-v2 --seed 20261016 --repeats 2 --margin 2.0
```

The baseline is PyTorch CPU fp32 versus C++ `hf_fp32`. Calibration uses only
calibration prompts, with three permitted C++ variants: default, pairwise dot
reductions, and FMA-free dots. `--margin` multiplies maximum benign per-op
tolerances and is recorded in `settings.json` (default 1.0; amended run: 2.0).
Maximum benign tolerances are frozen before any
localiser evaluation. Controls cycle these variants over every evaluation prompt.
Each bug type in the runtime registry is tested `repeats` times per layer on seeded
choices of evaluation prompts. Inputs were already inspected in M2: this split is
held out from tolerance calibration, **not previously unseen data**.

The executable and localiser source are copied before the run. Settings record
binary/source/input hashes, seed, versions, and the baseline. A sibling directory
`<output-name>-private/` holds injection records and backend logs. Keep it private and under
ignored `data/`. The localiser gets an opaque localhost replay URL. Only the
broker invokes the backend with a bug spec; neither the replay client nor the
localiser receives injection paths, labels or the seed. This is logical blinding,
not an OS sandbox against a malicious localiser.

All localiser reports are sealed before scoring reads ground truth. Effects are
classified independently by comparing each buggy backend against its clean C++
run under frozen tolerances. No-effect runs are reported separately and excluded
from the detection denominator. Detection means an op was flagged; localisation
also requires the correct first patched site. Benign alarms, backend/localiser
errors and no-effect alarms are all counted explicitly. Any execution error
prevents a passing summary.

Results include counts, rates, per-type coverage, and 10,000-resample percentile
intervals clustered by prompt ID across layers. Zero/all-success bootstrap
intervals may be degenerate; they do not prove a population error bound. Small
case counts and previously inspected inputs limit generalisation.

Outputs are reproducible research artifacts, not committed model data:
`settings.json`, `frozen.json`, `tolerances.json`, `sealed_reports.json`, per-case
reports, `scores.json`, and `summary.json`. No retuning occurs after evaluation.
