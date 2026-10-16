"""Load per-op dump directories (spec/INTERFACES.md section 2) and measure differences in units of the last place."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OPS = (
    "o_proj_out", "resid_sum", "router_input", "router_logits", "router_probs", "topk_ids", "topk_weights",
    "expert_gate", "expert_up", "expert_act", "expert_down", "combine_terms", "output",
)
SELECTION_OPS = OPS[:6]
SLOT_ALIGNED = ("topk_weights", "expert_gate", "expert_up", "expert_act", "expert_down", "combine_terms")

SPLITS = ("all", "calibration", "evaluation")
REPO = Path(__file__).resolve().parents[1]

# (explicit mantissa bits, exponent of the smallest normal)
_FORMATS = {"fp64": (52, -1022), "fp32": (23, -126), "bf16": (7, -126), "fp16": (10, -14)}
_ALIASES = {"float64": "fp64", "float32": "fp32", "bfloat16": "bf16", "float16": "fp16", "half": "fp16"}


def canonical_dtype(name: str) -> str:
    name = _ALIASES.get(name, name)
    if name not in _FORMATS:
        raise ValueError(f"unknown logical dtype: {name}")
    return name


def ulp(dtype: str, magnitude: np.ndarray) -> np.ndarray:
    """Spacing of `dtype` at |magnitude| (subnormal spacing below the normal range)."""
    mant, emin = _FORMATS[canonical_dtype(dtype)]
    a = np.abs(np.asarray(magnitude, dtype=np.float64))
    _, e = np.frexp(np.where(a > 0, a, 2.0 ** emin))
    exp = np.maximum(e.astype(np.int64) - 1, emin)
    return np.ldexp(1.0, (exp - mant).astype(np.int32))


@dataclass
class Run:
    path: Path
    manifest: dict
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    rows: np.ndarray | None = None  # token rows this view exposes (None = all); set by `select_split`
    split: str = "all"

    @property
    def tokens(self) -> int:
        return int(self.manifest["tokens"]) if self.rows is None else int(len(self.rows))

    @property
    def export_sha(self) -> str | None:
        return (self.manifest.get("export") or {}).get("config_sha256")

    def dtype(self, op: str) -> str:
        for o in self.manifest["ops"]:
            if o["name"] == op:
                return o["logical_dtype"]
        raise KeyError(op)

    def __getitem__(self, op: str) -> np.ndarray:
        if op not in self.arrays:
            entry = next(o for o in self.manifest["ops"] if o["name"] == op)
            a = np.load(self.path / entry["file"], allow_pickle=False)
            self.arrays[op] = a if self.rows is None else a[self.rows]
        return self.arrays[op]


def load_run(path: str | Path) -> Run:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    names = [o["name"] for o in manifest["ops"]]
    missing = [op for op in OPS if op not in names]
    if missing:
        raise ValueError(f"{path}: manifest lacks ops {missing}")
    return Run(path=path, manifest=manifest)


def _export_dir(run: Run) -> Path:
    p = Path((run.manifest.get("export") or {}).get("path", ""))
    for cand in (p, REPO / p, run.path / p):
        if (cand / "inputs" / "token_meta.json").exists():
            return cand
    raise FileNotFoundError(f"{run.path}: cannot find inputs/token_meta.json for export '{p}'")


def token_splits(run: Run) -> np.ndarray:
    """Split label of every token in the run (export's token_meta.json, sliced by the manifest's token_range)."""
    meta = json.loads((_export_dir(run) / "inputs" / "token_meta.json").read_text())
    labels = np.array([m["split"] for m in meta])
    a, b = run.manifest.get("token_range", [0, len(labels)])
    labels = labels[a:b]
    if len(labels) != int(run.manifest["tokens"]):
        raise ValueError(f"{run.path}: token_meta has {len(labels)} tokens in range, manifest says {run.manifest['tokens']}")
    return labels


def select_split(run: Run, split: str) -> Run:
    """A view of `run` restricted to one split's tokens ("all" returns the run unchanged)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "all":
        return run
    rows = np.flatnonzero(token_splits(run) == split)
    if len(rows) == 0:
        raise ValueError(f"{run.path}: no tokens in split '{split}'")
    return Run(path=run.path, manifest=run.manifest, rows=rows, split=split)


def check_same_export(ref: Run, cand: Run) -> None:
    if ref.export_sha and cand.export_sha and ref.export_sha != cand.export_sha:
        raise ValueError(f"runs come from different exports ({ref.export_sha[:12]} vs {cand.export_sha[:12]})")
    if ref.split != cand.split:
        raise ValueError(f"runs use different splits ({ref.split} vs {cand.split})")


def same_route_sets(ref_ids: np.ndarray, cand_ids: np.ndarray) -> np.ndarray:
    """[T] bool: the top-k sets are equal (order within the set ignored)."""
    return (np.sort(ref_ids, axis=1) == np.sort(cand_ids, axis=1)).all(axis=1)


def align_slots(ref_ids: np.ndarray, cand_ids: np.ndarray, cand: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Reorder the candidate's slot axis to the reference's expert order, on tokens whose sets match."""
    out = np.full_like(cand, np.nan, dtype=np.float64)
    for t in np.flatnonzero(same):
        pos = {int(e): s for s, e in enumerate(cand_ids[t])}
        out[t] = cand[t][[pos[int(e)] for e in ref_ids[t]]]
    return out
