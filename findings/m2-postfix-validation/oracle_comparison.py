"""Post-fix C++ port and PyTorch, each against the fp64 oracle (descriptive, post hoc).

Relative L2 (||x - ref|| / ||ref|| over the whole [tokens, ...] tensor) at expert_down and output, OLMoE layers 0 and 7,
all 1348 tokens: C++ hf_fp32 (post-fix dumps) vs the fp64 oracle, torch CPU fp32 vs the oracle, and C++ vs torch.
expert_down is slot-aligned by expert id (all route sets match in fp32). The oracle is the port's fp64 preset, run with
the post-fix binary port/build/moe_run into data/results/m2-postfix-oracle if its dump is missing.

  python findings/m2-postfix-validation/oracle_comparison.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from routediff.dumps import align_slots, same_route_sets  # noqa: E402

RES = ROOT / "data/results/m2-postfix-validation"  # frozen, read-only: only read from
ORACLE = ROOT / "data/results/m2-postfix-oracle"   # oracle dumps written here
MOE_RUN = ROOT / "port/build/moe_run"
OPS = ("expert_down", "output")


def oracle(layer: int) -> Path:
    d = ORACLE / f"layer{layer}"
    if not (d / "manifest.json").exists():
        ORACLE.mkdir(parents=True, exist_ok=True)
        num = ORACLE / "numerics_oracle.json"
        num.write_text(json.dumps({"preset": "oracle"}))
        subprocess.run([str(MOE_RUN), "--export", str(ROOT / f"data/export/olmoe-1b-7b-0924/layer{layer}"),
                        "--numerics", str(num), "--dump", str(d), "--ops", "topk_ids," + ",".join(OPS)],
                       check=True, capture_output=True)
    return d


def load(d: Path) -> dict:
    return {op: np.load(d / f"{op}.npy").astype(np.float64) for op in OPS} | {"ids": np.load(d / "topk_ids.npy")}


def aligned(ref: dict, x: dict, op: str) -> np.ndarray:
    if op != "expert_down":
        return x[op]
    same = same_route_sets(ref["ids"], x["ids"])
    assert same.all(), "fp32 route sets should all match"
    return align_slots(ref["ids"], x["ids"], x[op], same)


def rel_l2(ref: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


out = {"note": __doc__.split("\n\n")[0], "moe_run_sha256": hashlib.sha256(MOE_RUN.read_bytes()).hexdigest(), "layers": {}}
for L in (0, 7):
    o = load(oracle(L))
    c = load(RES / "dumps" / f"layer{L}-cpp-hf_fp32-postfix")
    t = load(ROOT / f"data/dumps/layer{L}-torch-float32")
    out["layers"][f"layer{L}"] = {
        op: {"cpp_vs_oracle": rel_l2(o[op], aligned(o, c, op)), "torch_vs_oracle": rel_l2(o[op], aligned(o, t, op)),
             "cpp_vs_torch": rel_l2(aligned(o, t, op), aligned(o, c, op))} for op in OPS}
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
for L, v in out["layers"].items():
    for op, r in v.items():
        print(f"{L} {op:12} C++ vs fp64 {r['cpp_vs_oracle']:.3e}  torch vs fp64 {r['torch_vs_oracle']:.3e}  C++ vs torch {r['cpp_vs_torch']:.3e}")
