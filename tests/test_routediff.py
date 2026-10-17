"""routediff end to end on a synthetic export, with the C++ port as the candidate backend.

These are not the blinded evaluation (that lives in evaluation/ and is scored separately). They check the mechanics:
identical runs are clean, every planted bug type is localised to the op it first changes, and calibrated tolerances
keep benign numeric differences from being flagged.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from routediff.compare import compare
from routediff.dumps import load_run, ulp
from routediff.localize import calibrate, localize
from routediff.synthetic import make_export

MOE_RUN = Path(__file__).resolve().parents[1] / "port" / "build" / "moe_run"
pytestmark = pytest.mark.skipif(not MOE_RUN.exists(), reason="build port/ first (make test-cpp)")

PARAMS = {
    "oproj_accum": {"dtype": "fp16"}, "rmsnorm_dtype": {"dtype": "bf16"}, "router_accum": {"dtype": "bf16"},
    "gate_out_dtype": {"dtype": "fp32"}, "softmax_dtype": {"dtype": "bf16"}, "weights_dtype": {"dtype": "fp32"},
    "expert_accum": {"dtype": "fp16"}, "combine_accum": {"dtype": "fp32"}, "combine_order": {"order": "expert_desc"},
}


def run(tmp: Path, name: str, export: Path, numerics: dict, bug: dict | None = None, extra: list[str] = ()) -> Path:
    num = tmp / f"{name}.numerics.json"
    num.write_text(json.dumps(numerics))
    cmd = [str(MOE_RUN), "--export", str(export), "--numerics", str(num), "--dump", str(tmp / name), *extra]
    if bug is not None:
        bp = tmp / f"{name}.bug.json"
        bp.write_text(json.dumps(bug))
        cmd += ["--bug", str(bp)]
    subprocess.run(cmd, check=True, capture_output=True)
    return tmp / name


def rerun_template(tmp: Path, name: str, export: Path) -> str:
    """What an evaluator would hand routediff: reruns the candidate (bug and all) with a patch."""
    bug = tmp / f"{name}.bug.json"
    tpl = f"{MOE_RUN} --export {export} --numerics {tmp / (name + '.numerics.json')} --dump {{out}} {{patch}}"
    return tpl + (f" --bug {bug}" if bug.exists() else "")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rd")
    export = make_export(tmp / "export", tokens=64, seed=3)
    ref = run(tmp, "ref", export, {"preset": "hf_bf16"})
    return tmp, export, ref


def test_ulp_matches_known_spacings():
    assert ulp("bf16", np.array([1.0]))[0] == 2 ** -7
    assert ulp("fp16", np.array([1.5]))[0] == 2 ** -10
    assert ulp("bf16", np.array([0.0]))[0] == 2.0 ** (-126 - 7)


def test_identical_runs_are_clean(setup):
    tmp, export, ref = setup
    again = run(tmp, "ref_again", export, {"preset": "hf_bf16"})
    res = compare(load_run(ref), load_run(again))
    assert res["flagged_ops"] == [] and res["route_mismatch_tokens"] == 0


def test_every_bug_type_is_localised_to_its_injected_op(setup):
    tmp, export, ref = setup
    registry = json.loads(subprocess.run([str(MOE_RUN), "--list-bugs"], check=True, capture_output=True,
                                         text=True).stdout)
    r = load_run(ref)
    ids = r["topk_ids"]
    results = {}
    for info in registry:
        t = info["type"]
        params = dict(PARAMS.get(t, {}))
        if t == "expert_permuted":
            params = {"a": str(int(ids[0, 0])), "b": str(int(ids[0, 1]))}
        name = f"bug_{t}"
        cand = run(tmp, name, export, {"preset": "hf_bf16"}, {"bug_id": name, "type": t, "params": params})
        rep = localize(r, load_run(cand), {}, rerun_template(tmp, name, export))
        results[t] = (rep["comparison"]["first_flagged_op"], rep["localized_op"], info["injected_op"])
    no_effect = {t for t, (first, _, _) in results.items() if first is None}
    assert no_effect <= {"topk_tiebreak_reversed"}, results  # needs an exact tie at the k-th place
    wrong = {t: v for t, v in results.items() if t not in no_effect and v[1] != v[2]}
    assert not wrong, wrong


def test_calibrated_tolerances_absorb_benign_differences(setup):
    # fp32 storage: on this small export the bf16 variants round to the same values as the reference (all-zero
    # tolerances), so the benign order / FMA differences are only visible in fp32.
    tmp, export, _ = setup
    r = load_run(run(tmp, "ref_fp32", export, {"preset": "hf_fp32"}))
    benign = [
        ("seq_nofma", {"preset": "hf_fp32", "expert_fma": "false", "router_fma": "false", "oproj_fma": "false"}),
        ("strided8", {"preset": "hf_fp32", "expert_order": "strided", "expert_block": 8,
                      "router_order": "strided", "router_block": 8, "oproj_order": "strided", "oproj_block": 8}),
        ("pairwise", {"preset": "hf_fp32", "expert_order": "pairwise", "router_order": "pairwise",
                      "oproj_order": "pairwise"}),
    ]
    runs = {n: load_run(run(tmp, n, export, num)) for n, num in benign}
    tol = calibrate([(r, runs["seq_nofma"]), (r, runs["strided8"])], margin=1.0)
    assert all(tol[op] > 0 for op in ("router_logits", "expert_gate", "output"))  # the benign variants do differ
    # A held-out benign pair stays far below a planted fp16 expert accumulator, which is localised.
    held = compare(r, runs["pairwise"], tol)
    bug = run(tmp, "cal_bug", export, {"preset": "hf_fp32"},
              {"bug_id": "x", "type": "expert_accum", "params": {"dtype": "fp16"}})
    rep = localize(r, load_run(bug), tol, rerun_template(tmp, "cal_bug", export))
    assert rep["localized_op"] == "expert_gate", rep["verdict"]
    worst_benign = max(held["ops"][op]["max_scaled_ulp"] for op in ("expert_gate", "output"))
    worst_bug = rep["comparison"]["ops"]["expert_gate"]["max_scaled_ulp"]
    assert worst_benign > 0 and worst_bug > 4 * worst_benign


# ---------------------------------------------------------------------------------------------- --split
def _clone_run(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _perturb_rows(run_dir: Path, op: str, rows: np.ndarray, scale: float = 1e-2) -> None:
    a = np.load(run_dir / f"{op}.npy")
    a[rows] = a[rows] * (1 + scale)
    np.save(run_dir / f"{op}.npy", a)


def test_split_labels_come_from_the_export(setup):
    from routediff.dumps import select_split, token_splits
    tmp, export, ref = setup
    labels = token_splits(load_run(ref))
    meta = json.loads((export / "inputs" / "token_meta.json").read_text())
    assert list(labels) == [m["split"] for m in meta]
    cal, ev = select_split(load_run(ref), "calibration"), select_split(load_run(ref), "evaluation")
    assert cal.tokens + ev.tokens == load_run(ref).tokens and cal.tokens > 0 and ev.tokens > 0
    assert cal["output"].shape[0] == cal.tokens


def test_calibration_never_touches_evaluation_tokens(setup):
    from routediff.dumps import select_split, token_splits
    tmp, export, ref = setup
    benign = run(tmp, "split_benign", export, {"preset": "hf_bf16", "expert_fma": "false", "router_fma": "false"})
    eval_rows = np.flatnonzero(token_splits(load_run(ref)) == "evaluation")
    noisy = _clone_run(benign, tmp / "split_benign_noisy_eval")
    for op in ("router_logits", "expert_gate", "output"):
        _perturb_rows(noisy, op, eval_rows)
    sel = lambda p: select_split(load_run(p), "calibration")
    tol_clean = calibrate([(sel(ref), sel(benign))])
    tol_noisy = calibrate([(sel(ref), sel(noisy))])
    strip = lambda t: {k: v for k, v in t.items() if k != "_meta"}
    assert strip(tol_clean) == strip(tol_noisy)
    assert tol_clean["_meta"]["split"] == "calibration"
    # and the perturbation is visible when evaluation tokens are included
    assert strip(calibrate([(load_run(ref), load_run(noisy))])) != strip(tol_clean)


def test_compare_on_evaluation_ignores_calibration_tokens(setup):
    from routediff.dumps import select_split, token_splits
    tmp, export, ref = setup
    cal_rows = np.flatnonzero(token_splits(load_run(ref)) == "calibration")
    noisy = _clone_run(ref, tmp / "ref_noisy_cal")
    _perturb_rows(noisy, "output", cal_rows)
    ev = lambda p: select_split(load_run(p), "evaluation")
    assert compare(ev(ref), ev(noisy))["flagged_ops"] == []
    assert compare(load_run(ref), load_run(noisy))["flagged_ops"] == ["output"]


def test_runs_from_different_exports_or_splits_are_rejected(setup):
    from routediff.dumps import select_split
    tmp, export, ref = setup
    other_export = make_export(tmp / "export_other", tokens=64, seed=99)
    other = run(tmp, "other_export_run", other_export, {"preset": "hf_bf16"})
    with pytest.raises(ValueError, match="different exports"):
        compare(load_run(ref), load_run(other))
    with pytest.raises(ValueError, match="different splits"):
        compare(select_split(load_run(ref), "calibration"), select_split(load_run(ref), "evaluation"))


def test_localize_judges_patch_checks_on_the_same_split(setup):
    from routediff.dumps import select_split
    tmp, export, ref = setup
    name = "split_bug"
    cand = run(tmp, name, export, {"preset": "hf_bf16"}, {"bug_id": name, "type": "router_accum", "params": {"dtype": "bf16"}})
    ev = lambda p: select_split(load_run(p), "evaluation")
    rep = localize(ev(ref), ev(cand), {}, rerun_template(tmp, name, export))
    assert rep["localized_op"] == "router_logits"
    assert rep["comparison"]["split"] == "evaluation"
    assert rep["comparison"]["tokens"] == ev(ref).tokens


# ---------------------------------------------------------------------------------------------- fidelity (H-port)
def test_normalised_margin_uses_kth_and_next_logit():
    from routediff.fidelity import normalised_margin
    logits = np.array([[4.0, 1.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]])
    m = normalised_margin(logits, 2)
    assert m[0] == pytest.approx(1.0 / np.std(logits[0]))  # 3rd-largest 2.0 vs 2nd-largest 3.0
    assert m[1] == 0.0  # constant logits: zero margin, no division by zero


def test_fidelity_pass_on_identical_runs_and_fail_on_ulp_drift(setup):
    from routediff.fidelity import ULP_P99, evaluate
    tmp, export, _ = setup
    ref = run(tmp, "fid_ref", export, {"preset": "hf_fp32"})
    same = evaluate(ref, _clone_run(ref, tmp / "fid_same"))
    assert same["verdict"] == "PASS" and same["failing_ops"] == [] and same["route_mismatch_tokens_all"] == 0
    drift = _clone_run(ref, tmp / "fid_drift")
    a = np.load(drift / "expert_down.npy")
    np.save(drift / "expert_down.npy", (a.astype(np.float64) * (1 + 2.0 ** -19)).astype(np.float32))  # ~16 fp32 ULP
    res = evaluate(ref, drift)
    assert res["verdict"] == "FAIL" and res["routes_pass"]
    assert res["failing_ops"] == ["expert_down"] and res["ops"]["expert_down"]["p99_elem_ulp"] > ULP_P99
    from routediff.fidelity import main
    assert main([str(ref), str(drift), "--out", str(tmp / "fid_drift.json")]) == 0  # the CLI writes valid JSON
    assert json.loads((tmp / "fid_drift.json").read_text())["verdict"] == "FAIL"


# ---------------------------------------------------------------------------------------------- non-finite values
def _with_value(src: Path, dst: Path, op: str, value: float, token: int = 3) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    _clone_run(src, dst)
    a = np.load(dst / f"{op}.npy")
    a.reshape(a.shape[0], -1)[token, 0] = value
    np.save(dst / f"{op}.npy", a)
    return dst


@pytest.fixture(scope="module")
def fp32_ref(setup):
    tmp, export, _ = setup
    return run(tmp, "nonfinite_ref", export, {"preset": "hf_fp32"})


@pytest.mark.parametrize("op", ["router_logits", "expert_down", "output"])
@pytest.mark.parametrize("side", ["candidate", "reference"])
def test_nan_in_either_run_is_flagged_never_exact(setup, fp32_ref, op, side):
    tmp, _, _ = setup
    nan = _with_value(fp32_ref, tmp / f"nan_{side}_{op}", op, np.nan)
    ref, cand = (fp32_ref, nan) if side == "candidate" else (nan, fp32_ref)
    res = compare(load_run(ref), load_run(cand))
    s = res["ops"][op]
    assert op in res["flagged_ops"] and not s["exact"] and s["max_scaled_ulp"] == np.inf
    assert s["compared_tokens"] == res["tokens"]  # NaN never removes a token from the comparison
    assert s["p99_elem_ulp"] < np.inf and s["max_elem_ulp"] == np.inf  # one bad element out of many


def test_infinities_equal_on_both_sides_match_one_sided_is_flagged(setup, fp32_ref):
    tmp, _, _ = setup
    both = _with_value(fp32_ref, tmp / "inf_ref", "output", np.inf)
    same = _with_value(fp32_ref, tmp / "inf_cand", "output", np.inf)
    assert compare(load_run(both), load_run(same))["flagged_ops"] == []
    assert compare(load_run(fp32_ref), load_run(same))["flagged_ops"] == ["output"]


def test_calibration_refuses_non_finite_benign_pairs(setup, fp32_ref):
    tmp, _, _ = setup
    nan = _with_value(fp32_ref, tmp / "nan_cal", "output", np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        calibrate([(load_run(fp32_ref), load_run(nan))])


def test_fidelity_fails_on_nan(setup, fp32_ref):
    from routediff.fidelity import evaluate
    tmp, _, _ = setup
    nan = _with_value(fp32_ref, tmp / "nan_fid", "expert_down", np.nan)
    res = evaluate(fp32_ref, nan)
    assert res["verdict"] == "FAIL" and res["failing_ops"] == ["expert_down"]  # one element: p99 alone would pass
    nan_logits = _with_value(fp32_ref, tmp / "nan_fid_ref", "router_logits", np.nan)
    assert evaluate(nan_logits, fp32_ref)["tokens_above_margin"] == load_run(fp32_ref).tokens  # NaN margin is checked


def test_route_mismatch_on_nan_probabilities_is_unexplained():
    from routediff.compare import _valid_topk
    probs = np.array([0.4, 0.3, np.nan, 0.1])
    assert _valid_topk(probs, np.array([0, 1]))
    assert not _valid_topk(np.array([0.4, 0.3, 0.2, 0.1]), np.array([0, 3]))
