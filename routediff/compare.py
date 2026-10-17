"""Op-by-op comparison of a candidate run against a reference run.

Float ops are measured in *scaled ULP*: |ref - cand| divided by the reference dtype's spacing at max(|ref|, row RMS).
Using the row RMS as a floor stops values that happen to sit near zero (where a tiny absolute error is many ULP) from
dominating. Plain element-wise ULP is reported as well (p99 / max), since PREREG's fidelity test is stated in it.

Non-finite values: at a compared position, a NaN in either run, or an infinity in only one of them, counts as an
infinite error (always over tolerance), so a backend that produces NaN is never reported as matching. Equal
infinities count as equal. Which tokens are compared at all (slot-aligned ops: tokens whose top-k sets match) is
decided by the route sets, never by NaN.

Top-k is compared as sets. A route mismatch at a token is *explained upstream* when the reference's set is not a valid
top-k of the candidate's own probabilities (the candidate genuinely ranks experts differently); it is *unexplained*
when the reference's set would also be a valid top-k of the candidate's probabilities, i.e. the selection step itself
chose differently (tie-break or selection logic).
"""
from __future__ import annotations

import numpy as np

from .dumps import OPS, SLOT_ALIGNED, Run, align_slots, check_same_export, same_route_sets, ulp


def _valid_topk(probs: np.ndarray, chosen: np.ndarray) -> bool:
    if not np.isfinite(probs).all():
        return True  # selection on non-finite probabilities is undefined: never call the mismatch explained upstream
    mask = np.zeros(probs.shape[0], dtype=bool)
    mask[chosen] = True
    return probs[mask].min() >= probs[~mask].max()


def route_stats(ref: Run, cand: Run) -> dict:
    rid, cid = ref["topk_ids"], cand["topk_ids"]
    same = same_route_sets(rid, cid)
    unexplained = [int(t) for t in np.flatnonzero(~same) if _valid_topk(cand["router_probs"][t], rid[t])]
    return {
        "tokens": int(len(same)),
        "route_mismatch_tokens": int((~same).sum()),
        "unexplained_tokens": len(unexplained),
        "explained_tokens": int((~same).sum()) - len(unexplained),
        "same": same,
    }


def _p99(x: np.ndarray) -> float:
    """np.percentile(x, 99) (linear), also when x holds +inf (numpy's interpolation would give NaN there)."""
    if np.isfinite(x).all():
        return float(np.percentile(x, 99))
    s = np.sort(x)
    pos = 0.99 * (len(s) - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if np.isinf(s[lo]) or (np.isinf(s[hi]) and pos > lo):
        return float("inf")
    return float(np.percentile(np.minimum(x, s[hi]), 99))  # the order statistics used are unchanged by the clip


def float_stats(ref: Run, cand: Run, op: str, same: np.ndarray) -> dict:
    a = ref[op].astype(np.float64)
    b = cand[op].astype(np.float64)
    T = a.shape[0]
    if op in SLOT_ALIGNED:
        b = align_slots(ref["topk_ids"], cand["topk_ids"], b, same)  # NaN-filled where sets differ
        compared = np.asarray(same, dtype=bool)
    else:
        compared = np.ones(T, dtype=bool)
    a2, b2 = a.reshape(T, -1), b.reshape(T, -1)
    b2 = np.where(compared[:, None], b2, a2)  # positions not compared contribute nothing below
    dt = ref.dtype(op)
    with np.errstate(invalid="ignore"):
        diff = np.where(a2 == b2, 0.0, np.abs(a2 - b2))  # equal infinities -> 0
    bad = np.isnan(a2) | np.isnan(b2) | (np.isinf(a2) != np.isinf(b2)) | (np.isinf(a2) & (a2 != b2))
    diff = np.where(bad, np.inf, diff)
    fa = np.where(np.isfinite(a2), a2, 0.0)  # magnitudes for the ULP scale ignore non-finite entries
    rms = np.sqrt(np.mean(fa ** 2, axis=1, keepdims=True))
    with np.errstate(invalid="ignore"):
        scaled = np.where(bad, np.inf, diff / ulp(dt, np.maximum(np.abs(fa), rms)))
        elem = np.where(bad, np.inf, diff / ulp(dt, fa))
    per_token = np.where(compared, scaled.max(axis=1), np.nan)
    v = compared
    elem_v = elem[v].ravel() if v.any() else np.zeros(1)
    num = np.sqrt(np.sum(diff[v] ** 2)) if v.any() else 0.0
    den = np.sqrt(np.sum(fa[v] ** 2)) if v.any() else 1.0
    return {
        "dtype": dt,
        "compared_tokens": int(v.sum()),
        "per_token_scaled_ulp": per_token,
        "max_scaled_ulp": float(np.nanmax(per_token)) if v.any() else 0.0,
        "p99_elem_ulp": _p99(elem_v),
        "max_elem_ulp": float(elem_v.max()),
        "rel_l2": float(num / den) if den > 0 else 0.0,
        "exact": bool(v.any() and np.array_equal(a2[v], b2[v])),
    }


def compare(ref: Run, cand: Run, tol: dict | None = None) -> dict:
    """Per-op stats and flags. `tol` maps op -> scaled-ULP tolerance and "topk_unexplained" -> allowed fraction.
    Without tolerances every difference is flagged (exact comparison)."""
    tol = tol or {}
    check_same_export(ref, cand)
    if ref.tokens != cand.tokens:
        raise ValueError("reference and candidate cover different numbers of tokens")
    routes = route_stats(ref, cand)
    same = routes["same"]
    ops: dict[str, dict] = {}
    for op in OPS:
        if op == "topk_ids":
            allowed = tol.get("topk_unexplained", 0.0) * routes["tokens"]
            ops[op] = {k: v for k, v in routes.items() if k != "same"}
            ops[op]["flagged"] = routes["unexplained_tokens"] > allowed
            continue
        s = float_stats(ref, cand, op, same)
        t = tol.get(op, 0.0)
        over = np.nan_to_num(s["per_token_scaled_ulp"], nan=0.0, posinf=np.inf) > t
        s["tolerance_scaled_ulp"] = t
        s["tokens_over_tolerance"] = int(over.sum())
        s["flagged"] = bool(over.any())
        s.pop("per_token_scaled_ulp")
        ops[op] = s
    flagged = [op for op in OPS if ops[op]["flagged"]]
    return {
        "tokens": ref.tokens,
        "split": ref.split,
        "route_mismatch_tokens": routes["route_mismatch_tokens"],
        "ops": ops,
        "flagged_ops": flagged,
        "first_flagged_op": flagged[0] if flagged else None,
    }


def per_token_scaled(ref: Run, cand: Run) -> dict[str, np.ndarray]:
    """Per-token scaled ULP for every float op (used by calibration)."""
    check_same_export(ref, cand)
    same = same_route_sets(ref["topk_ids"], cand["topk_ids"])
    return {op: float_stats(ref, cand, op, same)["per_token_scaled_ulp"] for op in OPS if op != "topk_ids"}
