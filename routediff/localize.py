"""Tolerance calibration on benign pairs, and patch-forward localisation (PREREG section 2).

Localisation (cumulative patch-forward): patch the first flagged op (replace its output with the reference's) and rerun
the candidate; if the reference is not yet restored, compare the rerun again and add the first op that is *still*
flagged, and repeat. Ops that only differed as a knock-on effect of an earlier site come back by themselves and are
never added, so the patched set is the list of places the fault touches (a bad accumulator can hit several matmuls at
once). The localised op is the first site. The reference counts as restored when:
  - selection op:      the reference top-k sets come back (route mismatches drop to the benign allowance);
                       if the candidate had no route mismatches to begin with, this is vacuous, so the combined
                       output criterion is used instead and the report says so;
  - post-selection op: the combined output comes back within its calibrated tolerance.

routediff never sees how the candidate was produced (bug specs stay with the evaluator). It only receives a rerun
command template containing `{out}` and `{patch}`, which it fills with an output directory and `--patch` arguments.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .compare import compare, per_token_scaled, route_stats
from .dumps import OPS, SELECTION_OPS, SLOT_ALIGNED, Run, load_run, select_split


def calibrate(pairs: list[tuple[Run, Run]], margin: float = 1.0) -> dict:
    """Per-op tolerance = margin x the largest per-token scaled ULP seen on benign pairs; topk_unexplained = the
    largest fraction of unexplained route mismatches seen. Also records the benign route-mismatch fraction."""
    worst: dict[str, float] = {op: 0.0 for op in OPS if op != "topk_ids"}
    unexplained = mismatch = 0.0
    for ref, cand in pairs:
        for op, v in per_token_scaled(ref, cand).items():
            if np.isfinite(v).any():
                worst[op] = max(worst[op], float(np.nanmax(v)))
        r = route_stats(ref, cand)
        unexplained = max(unexplained, r["unexplained_tokens"] / r["tokens"])
        mismatch = max(mismatch, r["route_mismatch_tokens"] / r["tokens"])
    tol = {op: margin * v for op, v in worst.items()}
    tol["topk_unexplained"] = unexplained
    tol["benign_route_mismatch"] = mismatch
    tol["_meta"] = {"pairs": len(pairs), "margin": margin, "split": pairs[0][0].split if pairs else "all",
                    "tokens_per_pair": [a.tokens for a, _ in pairs],
                    "pair_names": [f"{a.path.name} vs {b.path.name}" for a, b in pairs]}
    return tol


def _patch_args(ref: Run, ops: list[str]) -> list[str]:
    args: list[str] = []
    for op in ops:
        args += ["--patch", f"{op}={ref.path / (op + '.npy')}"]
    if any(op in SLOT_ALIGNED for op in ops):
        args += ["--patch-ids", str(ref.path / "topk_ids.npy")]
    return args


def rerun_with_patch(template: str, ref: Run, ops: list[str], workdir: Path) -> Run:
    """Rerun the candidate with `ops` patched from the reference (full token range), then restrict the result to the
    same split as the reference so patch checks are judged on the same tokens."""
    out = workdir / ("patched_" + "+".join(ops))
    cmd: list[str] = []
    for tok in shlex.split(template):
        if tok == "{patch}":
            cmd += _patch_args(ref, ops)
        else:
            cmd.append(tok.replace("{out}", str(out)))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"rerun failed for {ops}: {proc.stderr.strip()[-500:]}")
    return select_split(load_run(out), ref.split)


def _restored(ref: Run, rerun: Run, op: str, tol: dict, cand_route_mismatch: int, res: dict) -> dict:
    allowed = tol.get("benign_route_mismatch", 0.0) * res["tokens"]
    routes_back = res["route_mismatch_tokens"] <= allowed
    out = res["ops"]["output"]
    output_back = not out["flagged"]
    if op in SELECTION_OPS and cand_route_mismatch > allowed:
        criterion, ok = "route_sets", routes_back
    else:
        criterion, ok = "output_tolerance", output_back
    return {"patched": None, "criterion": criterion, "restored": bool(ok), "routes_restored": bool(routes_back),
            "output_restored": bool(output_back), "route_mismatch_after": res["route_mismatch_tokens"],
            "output_tokens_over_tolerance_after": out["tokens_over_tolerance"],
            "patch_skipped_token_ops": rerun.manifest.get("patch_skipped_token_ops", 0)}


def localize(ref: Run, cand: Run, tol: dict, rerun_template: str | None, max_sites: int = 13) -> dict:
    res = compare(ref, cand, tol)
    report = {"comparison": res, "patch_checks": [], "localized_op": None, "sites": [], "stage": None}
    if not res["flagged_ops"]:
        report["verdict"] = "no op exceeds its tolerance"
        return report
    if rerun_template is None:
        report["verdict"] = "flagged but not patch-checked (no rerun command given)"
        return report
    first = res["flagged_ops"][0]
    sites: list[str] = []
    still = res
    with tempfile.TemporaryDirectory(prefix="routediff_") as tmp:
        while len(sites) < max_sites:
            nxt = next((op for op in still["flagged_ops"] if op not in sites), None)
            if nxt is None:
                break
            sites.append(nxt)
            rerun = rerun_with_patch(rerun_template, ref, sites, Path(tmp))
            still = compare(ref, rerun, tol)
            check = _restored(ref, rerun, first, tol, res["route_mismatch_tokens"], still)
            check["patched"] = list(sites)
            check["still_flagged"] = still["flagged_ops"]
            report["patch_checks"].append(check)
            if check["restored"]:
                report["localized_op"] = sites[0]
                report["sites"] = list(sites)
                report["stage"] = "selection" if sites[0] in SELECTION_OPS else "post-selection"
                where = sites[0] if len(sites) == 1 else f"{sites[0]} (fault touches {', '.join(sites)})"
                report["verdict"] = f"localized to {where}: patching restores the reference ({check['criterion']})"
                return report
    report["verdict"] = "flagged ops found, but patching them did not restore the reference"
    return report


def save_tolerances(tol: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(tol, indent=2) + "\n")


def load_tolerances(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
