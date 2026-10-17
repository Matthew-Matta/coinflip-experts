"""Per-bug-type effects in the blinded real-v2 evaluation (descriptive; computed after all reports were sealed and scored).

Inputs: data/evaluation/real-v2/scores.json (released after scoring: case_id -> bug_type) and each case's sealed
routediff report (data/evaluation/real-v2/cases/<case_id>/report/report.json). Every report is first checked against
sealed_reports.json with the evaluator's own hash (evaluation/run.py:229: sha256 of json.dumps(report, sort_keys=True)).
Per bug type: route-set mismatches summed over the 4 cases / evaluation tokens summed over the 4 cases, and the median
over the 4 cases of the output-tensor relative L2 (one value per case: ||cand - ref|| / ||ref|| over the case's whole
output tensor, as routediff reports it). No private records or bug parameters are read.

  python findings/routediff-eval-real-v2/per_bug_type.py
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "data/evaluation/real-v2"
OUT = Path(__file__).with_suffix(".json")

scores = json.loads((EV / "scores.json").read_text())
sealed = json.loads((EV / "sealed_reports.json").read_text())
groups = defaultdict(list)
for c in scores:
    rep = json.loads((EV / "cases" / c["case_id"] / "report" / "report.json").read_text())
    h = hashlib.sha256(json.dumps(rep, sort_keys=True).encode()).hexdigest()
    seal = sealed[c["case_id"]]
    seal = seal if isinstance(seal, str) else seal.get("sha256", seal)
    if h != seal:
        raise SystemExit(f"seal mismatch for case {c['case_id']}")
    groups[c["bug_type"] if c["kind"] == "bug" else "benign"].append(rep)

table = {}
for t, reps in sorted(groups.items()):
    table[t] = {"cases": len(reps),
                "route_mismatch_tokens": sum(r["route_mismatch_tokens"] for r in reps),
                "tokens": sum(r["tokens"] for r in reps),
                "median_output_rel_l2": median(r["ops"]["output"]["rel_l2"] for r in reps)}
OUT.write_text(json.dumps({"note": __doc__.split("\n\n")[0], "seals_verified": sum(len(v) for v in groups.values()),
                           "by_type": table}, indent=2) + "\n")
for t, v in table.items():
    print(f"{t:24} {v['route_mismatch_tokens']:4}/{v['tokens']:4}  median output rel L2 {v['median_output_rel_l2']:.3g}")
