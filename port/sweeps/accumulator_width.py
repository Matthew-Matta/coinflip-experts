"""Accumulator-width sweep (EXPLORATORY, after preregistration): how many mantissa bits does each accumulator need?

For each accumulator of the fp32 port (o_proj, RMSNorm variance sum, router, expert matmuls, combine) and each width
m = 7..23 explicit mantissa bits (8-bit exponent, "e8m<m>": e8m7 rounds like bf16, e8m23 like fp32), only that one
accumulator is narrowed (as a sequential sum, the port's default order); everything else stays at the hf_fp32 preset,
except that every matmul that is not narrowed runs on the NEON fp32 path (4 accumulators), which is ~15x faster and
inside the calibrated benign envelope (findings/simd). Plus one Kahan row per op (compensated sum in e8m7, sequential
order). The base run is the same: hf_fp32 with all three matmul groups on NEON. Per configuration and layer:
  - vs the fp64 oracle: route flips (top-k set differs), router-logit relative L2 and max |error|, output relative L2;
  - routediff, calibrated: the frozen real-v2 tolerances against the torch fp32 reference on the evaluation split, with
    patch-forward reruns (does it detect, and where does it localise?);
  - routediff, strict: exact comparison against the base run (same binary), evaluation split: the first op that
    changes at all. For a matmul knob even e8m23 differs from the base there (sequential vs NEON order).
Dumps of each configuration are deleted after use (they are ~0.3 GB each); oracle and base dumps are kept.

  python -m port.sweeps.accumulator_width --export-root data/export/olmoe-1b-7b-0924 --layers 0 7 \
      --moe-run port/build-simd/moe_run --tolerances data/evaluation/real-v2/tolerances.json \
      --out data/results/accumulator-width
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coinflip.provenance import save_run  # noqa: E402
from routediff.compare import compare  # noqa: E402
from routediff.dumps import load_run, select_split  # noqa: E402
from routediff.localize import load_tolerances, localize  # noqa: E402

# accumulator -> (numerics key for the dtype, numerics key for Kahan, op where a change first shows)
KNOBS = {
    "oproj": ("oproj_accum", "oproj_kahan", "o_proj_out"),
    "rms": ("rms_accum", "rms_kahan", "router_input"),
    "router": ("router_accum", "router_kahan", "router_logits"),
    "expert": ("expert_accum", "expert_kahan", "expert_gate"),
    "combine": ("combine_accum", "combine_kahan", "output"),
}
MANTISSAS = tuple(range(7, 24))
MATMULS = ("oproj", "router", "expert")
BASE = {"preset": "hf_fp32", **{f"{k}_simd": True for k in MATMULS}}
METRIC_OPS = ("topk_ids", "router_logits", "output")
# The expert accumulator runs after selection and cannot flip this block's routes; its emulated matmuls make each
# patch-forward rerun slow, so calibrated localisation (reruns) is done only at these widths, detection at all.
EXPERT_LOCALIZE_AT = ("e8m7", "e8m12", "e8m16", "e8m20", "kahan_e8m7")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_moe(moe_run: Path, export: Path, numerics: dict, out: Path, threads: int, ops: str | None = None) -> float:
    out.mkdir(parents=True, exist_ok=True)
    nf = out.parent / f"{out.name}.numerics.json"
    nf.write_text(json.dumps(numerics))
    cmd = [str(moe_run), "--export", str(export), "--numerics", str(nf), "--dump", str(out), "--threads", str(threads)]
    if ops:
        cmd += ["--ops", ops]
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True)
    return time.time() - t0


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def oracle_metrics(cand: dict, oracle: dict, base: dict, rows: np.ndarray | None) -> dict:
    sel = (lambda x: x) if rows is None else (lambda x: x[rows])
    c, o, b = ({k: sel(v) for k, v in d.items()} for d in (cand, oracle, base))
    same_o = (np.sort(c["topk_ids"], 1) == np.sort(o["topk_ids"], 1)).all(1)
    same_b = (np.sort(c["topk_ids"], 1) == np.sort(b["topk_ids"], 1)).all(1)
    lerr = np.abs(c["router_logits"].astype(np.float64) - o["router_logits"].astype(np.float64))
    return {"tokens": int(len(same_o)), "flips_vs_oracle": int((~same_o).sum()), "flips_vs_base": int((~same_b).sum()),
            "router_logit_rel_l2_vs_oracle": rel_l2(o["router_logits"], c["router_logits"]),
            "router_logit_max_abs_err_vs_oracle": float(lerr.max()),
            "output_rel_l2_vs_oracle": rel_l2(o["output"], c["output"])}


def load_ops(d: Path) -> dict[str, np.ndarray]:
    return {op: np.load(d / f"{op}.npy") for op in METRIC_OPS}


def slim(rep: dict) -> dict:
    """A routediff report without the per-op tables, plus the per-op flags and worst scaled ULP vs tolerance."""
    c = rep["comparison"]
    return {"verdict": rep["verdict"], "localized_op": rep["localized_op"], "sites": rep["sites"],
            "first_flagged_op": c["first_flagged_op"], "flagged_ops": c["flagged_ops"],
            "route_mismatch_tokens": c["route_mismatch_tokens"], "tokens": c["tokens"], "split": c["split"],
            "patch_checks": rep["patch_checks"],
            "ops": {op: {k: v for k, v in s.items() if k in ("max_scaled_ulp", "tolerance_scaled_ulp",
                                                              "tokens_over_tolerance", "flagged", "rel_l2")}
                    for op, s in c["ops"].items() if op != "topk_ids"}}


def configs() -> list[tuple[str, str, dict]]:
    out = []
    for knob, (dkey, kkey, _) in KNOBS.items():
        base = dict(BASE)
        if knob in MATMULS:
            base[f"{knob}_simd"] = False  # the narrowed matmul is a sequential sum
        for m in MANTISSAS:
            out.append((knob, f"e8m{m}", {**base, dkey: f"e8m{m}"}))
        out.append((knob, "kahan_e8m7", {**base, dkey: "e8m7", kkey: "true"}))
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Per accumulator: the fewest mantissa bits m such that every width >= m gives 0 route flips vs the oracle on
    every layer (all tokens), and the detection floor of calibrated routediff (the widest m it still flags)."""
    out: dict = {"label": "EXPLORATORY (not preregistered); OLMoE-1B-7B layers " +
                 ", ".join(str(x) for x in sorted(df.layer.unique())) + "; fp32 port, one accumulator narrowed at a time"}
    plain = df[df.variant.str.startswith("e8m")].copy()
    plain["m"] = plain.variant.str[3:].astype(int)
    tokens = int(df[df.split == "all"].groupby("layer").tokens.first().sum())
    for knob in KNOBS:
        g = plain[(plain.knob == knob) & (plain.split == "all")]
        flips_by_m = g.groupby("m").flips_vs_oracle.sum().sort_index()
        need = None
        for m in sorted(flips_by_m.index, reverse=True):
            if flips_by_m[m] == 0:
                need = m
            else:
                break
        ge = g.groupby("m")
        det = g[g.calibrated_flagged].m
        loc_ok = g[g.calibrated_localized_op == KNOBS[knob][2]].m
        checked = g[g.calibrated_patch_checked].m
        k = df[(df.knob == knob) & (df.variant == "kahan_e8m7") & (df.split == "all")]
        out[knob] = {
            "flips_vs_oracle_by_m": {int(m): int(v) for m, v in flips_by_m.items()},
            "router_logit_rel_l2_by_m": {int(m): float(v) for m, v in ge.router_logit_rel_l2_vs_oracle.max().items()},
            "output_rel_l2_by_m": {int(m): float(v) for m, v in ge.output_rel_l2_vs_oracle.max().items()},
            "min_bits_for_zero_flips": need,
            "tokens": tokens,
            "calibrated_detects_at_m": sorted(int(x) for x in set(det)),
            "calibrated_detection_floor_m": int(det.max()) if len(det) else None,
            "calibrated_localizes_correctly_at_m": sorted(int(x) for x in set(loc_ok)),
            "calibrated_patch_checked_at_m": sorted(int(x) for x in set(checked)),
            "strict_first_op_correct_all_m_below_23": bool((g[g.m < 23].strict_first_flagged_op == KNOBS[knob][2]).all()),
            "strict_first_op_at_m23": sorted(set(g[g.m == 23].strict_first_flagged_op.fillna("none"))),
            "kahan_e8m7": {"flips_vs_oracle": int(k.flips_vs_oracle.sum()),
                           "router_logit_rel_l2": float(k.router_logit_rel_l2_vs_oracle.max()),
                           "output_rel_l2": float(k.output_rel_l2_vs_oracle.max()),
                           "calibrated_flagged": bool(k.calibrated_flagged.any())},
        }
        if need is not None:
            out[knob]["sentence"] = (f"{knob} accumulator needs >= {need} mantissa bits (8-bit exponent) for 0 route flips "
                                     f"vs the fp64 oracle on {tokens} tokens"
                                     + (" (narrower widths never flip in this range)" if need == MANTISSAS[0] else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-root", required=True, type=Path)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 7])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--moe-run", type=Path, default=REPO / "port" / "build-simd" / "moe_run")
    ap.add_argument("--tolerances", type=Path, required=True)
    ap.add_argument("--torch-ref", default="data/dumps/layer{layer}-torch-float32")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="restrict to these accumulators (default: all)")
    a = ap.parse_args(argv)
    if (a.out / "accumulator_width.parquet").exists():
        ap.error(f"{a.out} already holds results; use a new --out")
    tol = load_tolerances(a.tolerances)
    t_start = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    rows_path = a.out / "rows.jsonl"  # one line per finished (layer, config, split): lets an interrupted run resume
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["layer"], r["knob"], r["variant"]) for r in rows}
    for layer in a.layers:
        export = a.export_root / f"layer{layer}"
        rd = a.out / "runs" / f"layer{layer}"
        if not (rd / "oracle" / "manifest.json").exists():
            run_moe(a.moe_run, export, {"preset": "oracle"}, rd / "oracle", a.threads, ",".join(METRIC_OPS))
        if not (rd / "base" / "manifest.json").exists():
            run_moe(a.moe_run, export, BASE, rd / "base", a.threads)
        oracle, base = load_ops(rd / "oracle"), load_ops(rd / "base")
        base_run = load_run(rd / "base")
        eval_rows = select_split(base_run, "evaluation").rows
        torch_ref = select_split(load_run(REPO / a.torch_ref.format(layer=layer)), "evaluation")
        base_eval = select_split(base_run, "evaluation")
        for knob, variant, num in configs():
            if (a.only and knob not in a.only) or (layer, knob, variant) in done:
                continue
            d = rd / f"{knob}-{variant}"
            secs = run_moe(a.moe_run, export, num, d, a.threads)
            cand_ops = load_ops(d)
            cand = select_split(load_run(d), "evaluation")
            tpl = (f"{a.moe_run} --export {export} --numerics {rd / (d.name + '.numerics.json')} "
                   f"--dump {{out}} {{patch}} --threads {a.threads}")
            t1 = time.time()
            rerun = tpl if knob != "expert" or variant in EXPERT_LOCALIZE_AT else None
            cal = slim(localize(torch_ref, cand, tol, rerun))
            strict = compare(base_eval, cand)
            rd_secs = time.time() - t1
            rep_dir = a.out / "routediff" / f"layer{layer}"
            rep_dir.mkdir(parents=True, exist_ok=True)
            (rep_dir / f"{knob}-{variant}.json").write_text(json.dumps(
                {"numerics": num, "calibrated": cal,
                 "strict_vs_cpp_base": {"first_flagged_op": strict["first_flagged_op"],
                                        "flagged_ops": strict["flagged_ops"],
                                        "route_mismatch_tokens": strict["route_mismatch_tokens"]}}, indent=1) + "\n")
            common = {"layer": layer, "knob": knob, "variant": variant, "first_op": KNOBS[knob][2],
                      "calibrated_flagged": bool(cal["flagged_ops"]), "calibrated_first_flagged_op": cal["first_flagged_op"],
                      "calibrated_localized_op": cal["localized_op"], "calibrated_sites": ",".join(cal["sites"]),
                      "strict_first_flagged_op": strict["first_flagged_op"], "seconds": secs, "routediff_seconds": rd_secs}
            common["calibrated_patch_checked"] = rerun is not None
            for split, r in (("all", None), ("evaluation", eval_rows)):
                rows.append({**common, "split": split, **oracle_metrics(cand_ops, oracle, base, r)})
                with rows_path.open("a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
            m = rows[-2]
            print(f"layer {layer} {knob:7} {variant:10}: flips vs oracle {m['flips_vs_oracle']:3}/{m['tokens']} "
                  f"logit relL2 {m['router_logit_rel_l2_vs_oracle']:.2e} out relL2 {m['output_rel_l2_vs_oracle']:.2e} | "
                  f"routediff cal: {cal['localized_op'] or cal['first_flagged_op'] or '-'}; strict first: "
                  f"{strict['first_flagged_op'] or '-'}  [{secs:.0f}+{rd_secs:.0f}s]", flush=True)
            shutil.rmtree(d)
    df = pd.DataFrame(rows)
    config = {"experiment": "accumulator_width", "status": "exploratory (post-preregistration)",
              "layers": a.layers, "mantissas": list(MANTISSAS), "knobs": {k: v[0] for k, v in KNOBS.items()},
              "kahan": "sequential Kahan sum in e8m7, one row per accumulator",
              "base": BASE, "narrowed_accumulator_order": "sequential (rms: torch_cascade as in hf_fp32; Kahan rows sequential)",
              "moe_run_sha256": sha(a.moe_run), "tolerances": str(a.tolerances), "tolerances_sha256": sha(a.tolerances),
              "torch_ref": a.torch_ref, "export_config_sha256": {L: sha(a.export_root / f"layer{L}" / "config.json")
                                                                  for L in a.layers},
              "wall_seconds": round(time.time() - t_start, 1)}
    pq = save_run(df, a.out, config, name="accumulator_width")
    summary = summarize(df)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {pq}")
    for knob in KNOBS:
        if knob in summary:
            print(summary[knob].get("sentence", f"{knob}: flips at every width up to 23 bits"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
