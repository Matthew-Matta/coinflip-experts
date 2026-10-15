"""report.json (for the evaluator) and report.md (for people), per spec/INTERFACES.md section 4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dumps import OPS, Run


def write_reports(result: dict, ref: Run, cand: Run, out_dir: str | Path, tol_path: str | None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comp = result["comparison"]
    doc = {
        "reference_run": str(ref.path),
        "candidate_run": str(cand.path),
        "tokens": comp["tokens"],
        "route_mismatch_tokens": comp["route_mismatch_tokens"],
        "ops": comp["ops"],
        "flagged_ops": comp["flagged_ops"],
        "first_flagged_op": comp["first_flagged_op"],
        "localized_op": result["localized_op"],
        "sites": result.get("sites", []),
        "stage": result["stage"],
        "patch_checks": result["patch_checks"],
        "verdict": result["verdict"],
        "tolerances": None if tol_path is None else
        {"file": tol_path, "sha256": hashlib.sha256(Path(tol_path).read_bytes()).hexdigest()},
    }
    (out / "report.json").write_text(json.dumps(doc, indent=2) + "\n")

    lines = [
        "# routediff report", "",
        f"- Reference: `{ref.path}` ({ref.manifest.get('backend', '?')})",
        f"- Candidate: `{cand.path}` ({cand.manifest.get('backend', '?')})",
        f"- Tokens: {comp['tokens']}; tokens whose expert set differs: {comp['route_mismatch_tokens']}",
        "", f"**Verdict:** {result['verdict']}", "",
        "| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, op in enumerate(OPS, 1):
        s = comp["ops"][op]
        mark = "**yes**" if s["flagged"] else ""
        if op == "topk_ids":
            lines.append(f"| {i} | `{op}` | {mark} | set mismatches: {s['route_mismatch_tokens']} "
                         f"(unexplained {s['unexplained_tokens']}) | | | | |")
        else:
            lines.append(f"| {i} | `{op}` | {mark} | {s['max_scaled_ulp']:.3g} | {s['tolerance_scaled_ulp']:.3g} | "
                         f"{s['tokens_over_tolerance']} | {s['p99_elem_ulp']:.3g} | {s['rel_l2']:.2e} |")
    if result["patch_checks"]:
        lines += ["", "## Patch checks (cumulative)", "",
                  "| patched so far | criterion | restored | routes back | output back | still flagged |",
                  "|---|---|---|---|---|---|"]
        for c in result["patch_checks"]:
            lines.append(f"| {', '.join('`' + o + '`' for o in c['patched'])} | {c['criterion']} | {c['restored']} | "
                         f"{c['routes_restored']} | {c['output_restored']} | {', '.join(c['still_flagged']) or '-'} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return out
