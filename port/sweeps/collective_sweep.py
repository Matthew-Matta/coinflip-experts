"""H4 collective sweep (PREREG section 3, emulation): wire dtype x topology x emulated chip count, on the real layer.

For each layer, every configuration is run with `moe_run` on the base numerics (`--base`, default `hf_fp32`, the
primary configuration per the 2026-10-16 PREREG amendment; `hf_bf16` is exploratory) and compared with the fp64 oracle:
  - target "oproj": the row-parallel attention output projection is all-reduced across n emulated chips; its result
    becomes the router input, so the metric is the route-flip rate (fraction of tokens whose top-k set differs);
  - target "down":  the row-parallel expert down projection is all-reduced; it cannot change this block's routes, so
    the metric is combined-output error (relative L2, p99 element ULP in the base's storage format: fp32 units for
    hf_fp32, bf16 units for hf_bf16), never counted as flips.
n = 1 (no collective) is the baseline. Everything here is emulation, not a measurement of any interconnect.

  python -m port.sweeps.collective_sweep --export-root data/export/olmoe-1b-7b-0924 --layers 0 7 --base hf_fp32 \
      --out data/results/h4-collective-sweep-fp32
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coinflip.provenance import save_run  # noqa: E402
from routediff.dumps import ulp  # noqa: E402

NS = (2, 4, 8, 16, 32, 64)
TOPOLOGIES = ("ring", "tree")
WIRES = ("bf16", "fp32")
TARGETS = ("oproj", "down")
OPS = "topk_ids,router_logits,output"
BASES = {"hf_fp32": "fp32", "hf_bf16": "bf16"}  # base preset -> storage dtype used for output ULP


def run_moe(moe_run: Path, export: Path, numerics: dict, out: Path, threads: int) -> float:
    out.mkdir(parents=True, exist_ok=True)
    nf = out / "numerics.json"
    nf.write_text(json.dumps(numerics))
    t0 = time.time()
    cmd = [str(moe_run), "--export", str(export), "--numerics", str(nf), "--dump", str(out), "--ops", OPS]
    if threads:
        cmd += ["--threads", str(threads)]
    subprocess.run(cmd, check=True, capture_output=True)
    return time.time() - t0


def load(d: Path) -> dict[str, np.ndarray]:
    return {op: np.load(d / f"{op}.npy") for op in OPS.split(",")}


def metrics(run: dict[str, np.ndarray], oracle: dict[str, np.ndarray], base: dict[str, np.ndarray],
            ulp_dtype: str) -> dict:
    same_oracle = (np.sort(run["topk_ids"], 1) == np.sort(oracle["topk_ids"], 1)).all(1)
    same_base = (np.sort(run["topk_ids"], 1) == np.sort(base["topk_ids"], 1)).all(1)
    o, c = oracle["output"].astype(np.float64), run["output"].astype(np.float64)
    elem = np.abs(o - c) / ulp(ulp_dtype, o)
    return {
        "tokens": int(len(same_oracle)),
        "flips_vs_oracle": int((~same_oracle).sum()),
        "flip_rate_vs_oracle": float((~same_oracle).mean()),
        "flips_vs_n1": int((~same_base).sum()),
        "output_rel_l2_vs_oracle": float(np.linalg.norm(o - c) / np.linalg.norm(o)),
        "output_p99_ulp_vs_oracle": float(np.percentile(elem, 99)),
        "ulp_dtype": ulp_dtype,
    }


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_h4(df: pd.DataFrame) -> dict:
    """PREREG H4 on the router-input (oproj) all-reduce, per layer: (a) bf16 ring flip rate vs oracle rises with n
    (Spearman rho >= 0.5 over n in NS); (b) fp32 flip rate at every n within 2x its n = 2 level, or, if that level is
    zero, below the rule-of-three bound 3 / tokens. (b) is checked for both topologies."""
    out = {}
    o = df[df.target == "oproj"]
    for layer, g in o.groupby("layer"):
        ring_bf16 = g[(g.topology == "ring") & (g.wire == "bf16") & (g.n >= 2)].sort_values("n")
        rho = spearman(ring_bf16.n.tolist(), ring_bf16.flip_rate_vs_oracle.tolist())
        a_pass = bool(rho >= 0.5)
        b = {}
        for topo in TOPOLOGIES:
            f = g[(g.topology == topo) & (g.wire == "fp32") & (g.n >= 2)].sort_values("n")
            level2 = float(f[f.n == 2].flip_rate_vs_oracle.iloc[0])
            tokens = int(f.tokens.iloc[0])
            bound = 2 * level2 if level2 > 0 else 3 / tokens
            ok = bool((f.flip_rate_vs_oracle <= bound).all()) if level2 > 0 else bool((f.flip_rate_vs_oracle < bound).all())
            b[topo] = {"n2_level": level2, "bound": bound, "rule": "2x n=2" if level2 > 0 else "rule of three",
                       "rates": dict(zip(f.n.tolist(), f.flip_rate_vs_oracle.tolist())), "pass": ok}
        b_pass = all(v["pass"] for v in b.values())
        out[f"layer{layer}"] = {
            "a_bf16_ring": {"spearman_rho": rho, "rates": dict(zip(ring_bf16.n.tolist(), ring_bf16.flip_rate_vs_oracle.tolist())),
                            "pass": a_pass},
            "b_fp32": b,
            "verdict": "PASS" if a_pass and b_pass else "FAIL",
        }
    out["verdict_all_layers"] = "PASS" if all(v["verdict"] == "PASS" for k, v in out.items() if k.startswith("layer")) else "FAIL"
    out["label"] = "emulation (PREREG section 3); not a measurement of any interconnect"
    out["base_numerics"] = df.attrs.get("base", "unknown")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export-root", required=True, type=Path)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 7])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--moe-run", type=Path, default=REPO / "port" / "build" / "moe_run")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ns", nargs="+", type=int, default=list(NS))
    ap.add_argument("--base", choices=sorted(BASES), default="hf_fp32",
                    help="hf_fp32 = primary (PREREG amendment 2026-10-16); hf_bf16 = exploratory")
    a = ap.parse_args(argv)
    if (a.out / "h4_collective_sweep.parquet").exists() or (a.out / "runs").exists():
        ap.error(f"{a.out} already holds sweep results; use a new --out directory")
    ulp_dtype = BASES[a.base]

    runs_dir = a.out / "runs"
    rows = []
    t_start = time.time()
    for layer in a.layers:
        export = a.export_root / f"layer{layer}"
        oracle_dir, base_dir = runs_dir / f"layer{layer}" / "oracle", runs_dir / f"layer{layer}" / "n1"
        secs = run_moe(a.moe_run, export, {"preset": "oracle"}, oracle_dir, a.threads)
        print(f"layer {layer}: oracle in {secs:.0f}s", flush=True)
        secs = run_moe(a.moe_run, export, {"preset": a.base}, base_dir, a.threads)
        oracle, base = load(oracle_dir), load(base_dir)
        m = metrics(base, oracle, base, ulp_dtype)
        for target in TARGETS:  # n = 1 is shared by every topology / wire; record it once per target
            rows.append({"layer": layer, "target": target, "n": 1, "topology": "none", "wire": "none", "seconds": secs, **m})
        print(f"layer {layer}: n=1 baseline flips vs oracle {m['flips_vs_oracle']}/{m['tokens']}", flush=True)
        for target, n, topo, wire in itertools.product(TARGETS, a.ns, TOPOLOGIES, WIRES):
            num = {"preset": a.base, "collective_n": n, "collective_topology": topo, "collective_wire": wire,
                   "collective_on": target}
            d = runs_dir / f"layer{layer}" / f"{target}-n{n}-{topo}-{wire}"
            secs = run_moe(a.moe_run, export, num, d, a.threads)
            m = metrics(load(d), oracle, base, ulp_dtype)
            rows.append({"layer": layer, "target": target, "n": n, "topology": topo, "wire": wire, "seconds": secs, **m})
            print(f"layer {layer} {target:5} n={n:2} {topo:4} {wire}: flips vs oracle {m['flips_vs_oracle']:4} "
                  f"(vs n1 {m['flips_vs_n1']:3})  out p99 {m['output_p99_ulp_vs_oracle']:.3g} {ulp_dtype} ulp  [{secs:.0f}s]",
                  flush=True)

    df = pd.DataFrame(rows)
    df.attrs["base"] = a.base
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config = {"experiment": "h4_collective_sweep", "base_numerics": a.base, "ulp_dtype": ulp_dtype,
              "status": "primary" if a.base == "hf_fp32" else "exploratory", "layers": a.layers, "ns": a.ns,
              "topologies": TOPOLOGIES, "wires": WIRES, "targets": TARGETS, "moe_run_sha256": sha(a.moe_run),
              "export_config_sha256": {L: sha(a.export_root / f"layer{L}" / "config.json") for L in a.layers},
              "wall_seconds": round(time.time() - t_start, 1), "label": "emulation"}
    pq = save_run(df, a.out, config, name="h4_collective_sweep")
    h4 = evaluate_h4(df)
    (a.out / "h4_verdict.json").write_text(json.dumps(h4, indent=2) + "\n")
    print(f"\nwrote {pq}\nH4: " + ", ".join(f"{k} {v['verdict']}" for k, v in h4.items() if k.startswith("layer"))
          + f" -> {h4['verdict_all_layers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
