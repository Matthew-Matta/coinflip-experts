"""PREREG section 1 (H-port): C++ block fidelity against the PyTorch reference, evaluated exactly as preregistered.

H-port: in fp32, the C++ block selects the same top-k set as PyTorch on every captured token whose normalised margin
exceeds MARGIN, and every intermediate op is within ULP_P99 ULP at the 99th percentile. Falsified if either fails.

  python -m routediff.fidelity REF_DUMP CAND_DUMP --out metrics.json

Thresholds are the preregistered ones and are not parameters on purpose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .compare import compare
from .dumps import OPS, load_run, same_route_sets

MARGIN = 1e-4   # PREREG section 1: normalised-margin cutoff
ULP_P99 = 8.0   # PREREG section 1: p99 element ULP bound per op


def normalised_margin(logits: np.ndarray, k: int) -> np.ndarray:
    """(logit_k - logit_(k+1)) / std(logits) per token, from the reference's router logits."""
    s = np.sort(logits.astype(np.float64), axis=1)[:, ::-1]
    std = logits.astype(np.float64).std(axis=1)
    return (s[:, k - 1] - s[:, k]) / np.where(std > 0, std, 1.0)


def evaluate(ref_dir: str | Path, cand_dir: str | Path) -> dict:
    ref, cand = load_run(ref_dir), load_run(cand_dir)
    k = ref["topk_ids"].shape[1]
    margin = normalised_margin(ref["router_logits"], k)
    same = same_route_sets(ref["topk_ids"], cand["topk_ids"])
    above = ~(margin <= MARGIN)  # a non-finite margin (NaN logits) is checked, never skipped
    res = compare(ref, cand)
    ops = {op: {"p99_elem_ulp": res["ops"][op]["p99_elem_ulp"], "max_elem_ulp": res["ops"][op]["max_elem_ulp"],
                "dtype": res["ops"][op]["dtype"],
                # a NaN / one-sided Inf anywhere fails the op even if it is under 1% of elements
                "pass": bool(res["ops"][op]["p99_elem_ulp"] <= ULP_P99 and np.isfinite(res["ops"][op]["max_elem_ulp"]))}
           for op in OPS if op != "topk_ids"}
    routes_pass = bool(same[above].all())
    return {
        "hypothesis": "H-port (PREREG section 1)",
        "thresholds": {"normalised_margin": MARGIN, "p99_elem_ulp": ULP_P99},
        "reference": {"path": str(ref.path), "manifest_sha256": _sha(ref.path / "manifest.json"),
                      "numerics": ref.manifest.get("numerics")},
        "candidate": {"path": str(cand.path), "manifest_sha256": _sha(cand.path / "manifest.json"),
                      "numerics": cand.manifest.get("numerics"), "provenance": cand.manifest.get("provenance")},
        "tokens": int(len(same)),
        "tokens_above_margin": int(above.sum()),
        "route_mismatch_tokens_above_margin": int((~same & above).sum()),
        "route_mismatch_tokens_all": int((~same).sum()),
        "routes_pass": routes_pass,
        "ops": ops,
        "ulp_pass": all(o["pass"] for o in ops.values()),
        "failing_ops": [op for op, o in ops.items() if not o["pass"]],
        "verdict": "PASS" if routes_pass and all(o["pass"] for o in ops.values()) else "FAIL",
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="routediff.fidelity", description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = evaluate(a.reference, a.candidate)
    Path(a.out).write_text(json.dumps(m, indent=2) + "\n")
    print(f"{m['verdict']}: routes {'pass' if m['routes_pass'] else 'FAIL'} "
          f"({m['route_mismatch_tokens_above_margin']} mismatches above margin, {m['tokens_above_margin']} tokens); "
          f"failing ops: {', '.join(m['failing_ops']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
