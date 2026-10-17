"""Supplementary (EXPLORATORY): bf16 output tolerance and verdicts on route-matched tokens only.

routediff compares `output` on every token, including tokens whose top-k sets differ. In bf16 those are mostly exact
ties decided differently (C++ lower id vs torch CPU nth_element), and their outputs differ completely, which inflates
the calibrated output tolerance. Here the same recipe (margin 2.0, calibration split) is applied to route-matched tokens.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from routediff.compare import float_stats  # noqa: E402
from routediff.dumps import load_run, same_route_sets, select_split  # noqa: E402

DUMPS = REPO / "data/results/bf16-tolerances"  # C++ calibration-variant dumps written by run.sh
MARGIN = 2.0
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", type=Path, default=Path(__file__).with_name("matched_only.json"))
OUT = ap.parse_args().out


def matched_worst(ref, cand):
    same = same_route_sets(ref["topk_ids"], cand["topk_ids"])
    per = float_stats(ref, cand, "output", same)["per_token_scaled_ulp"]
    return float(np.nanmax(per[same])), float(np.nanmax(per)), int((~same).sum()), int(len(same))


res = {"calibration": {}, "judged_evaluation": {}}
worst = 0.0
for L in (0, 7):
    ref = select_split(load_run(REPO / f"data/dumps/layer{L}-torch-bfloat16"), "calibration")
    for n in range(3):
        m, a, mis, tok = matched_worst(ref, select_split(load_run(DUMPS / f"layer{L}-cpp-bf16-cal-{n}"), "calibration"))
        res["calibration"][f"layer{L}-cal-{n}"] = {"output_worst_matched": m, "output_worst_all": a,
                                                   "route_mismatch": mis, "tokens": tok}
        worst = max(worst, m)
tol = MARGIN * worst
res["output_tolerance_matched_only"] = tol
res["output_tolerance_routediff_all_tokens"] = json.loads(
    (REPO / "data/evaluation/tolerances-bf16-exploratory.json").read_text())["output"]
for L in (0, 7):
    ref = select_split(load_run(REPO / f"data/dumps/layer{L}-torch-bfloat16"), "evaluation")
    cands = {"mps-eager": REPO / f"data/dumps/real-backend-pairs-v1/layer{L}-mps-eager-bfloat16",
             "cpu-grouped_mm": REPO / f"data/dumps/real-backend-pairs-v1/layer{L}-cpu-grouped_mm-bfloat16",
             **{f"cpp-cal-{n}": DUMPS / f"layer{L}-cpp-bf16-cal-{n}" for n in range(3)}}
    for name, p in cands.items():
        m, a, mis, tok = matched_worst(ref, select_split(load_run(p), "evaluation"))
        res["judged_evaluation"][f"layer{L}-{name}"] = {"output_worst_matched": m, "ratio_to_matched_tol": m / tol,
                                                        "flagged_matched_only": m > tol, "route_mismatch": mis,
                                                        "tokens": tok}
OUT.write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "calibration"}, indent=1))
