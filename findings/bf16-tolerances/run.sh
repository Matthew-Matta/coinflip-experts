#!/usr/bin/env bash
# EXPLORATORY bf16 tolerances (post-preregistration): the real-v2 calibration recipe (evaluation/run.py:29-31: default,
# pairwise and FMA-free C++ variants vs the torch CPU reference; margin 2.0 per PREREG 5dc67ef) applied to bf16:
# torch CPU bf16 reference vs C++ hf_bf16 variants, calibration split only, both layers. Then the bf16 real backend
# pairs (MPS eager, CPU grouped_mm) are judged against the torch CPU bf16 reference on the evaluation split (no rerun:
# torch backends cannot be patched).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT=data/results/bf16-tolerances; M=port/build/moe_run; PY=.venv/bin/python
TOL=data/evaluation/tolerances-bf16-exploratory.json
[ -e "$TOL" ] && { echo "$TOL exists; refusing to overwrite"; exit 1; }
echo '{"preset": "hf_bf16"}' > $OUT/cal-0.json
echo '{"preset": "hf_bf16", "oproj_order": "pairwise", "router_order": "pairwise", "expert_order": "pairwise"}' > $OUT/cal-1.json
echo '{"preset": "hf_bf16", "oproj_fma": "false", "router_fma": "false", "expert_fma": "false"}' > $OUT/cal-2.json
pairs=()
for L in 0 7; do
  for n in 0 1 2; do
    $M --export data/export/olmoe-1b-7b-0924/layer$L --numerics $OUT/cal-$n.json --dump $OUT/layer$L-cpp-bf16-cal-$n --threads 8
    pairs+=(data/dumps/layer$L-torch-bfloat16 $OUT/layer$L-cpp-bf16-cal-$n)
  done
done
$PY -m routediff calibrate "${pairs[@]}" --out $TOL --margin 2.0 --split calibration
shasum -a 256 $TOL $M > $OUT/sha256.txt
for L in 0 7; do
  for b in mps-eager cpu-grouped_mm; do
    $PY -m routediff compare data/dumps/layer$L-torch-bfloat16 data/dumps/real-backend-pairs-v1/layer$L-$b-bfloat16 \
        --out $OUT/report-layer$L-$b-bf16tol --tolerances $TOL --split evaluation
  done
  # benign sanity: the three C++ calibration variants judged on the held-out evaluation split
  for n in 0 1 2; do
    $PY -m routediff compare data/dumps/layer$L-torch-bfloat16 $OUT/layer$L-cpp-bf16-cal-$n \
        --out $OUT/report-layer$L-cpp-bf16-cal-$n-heldout --tolerances $TOL --split evaluation
  done
done
