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
    tmp, export, ref = setup
    r = load_run(ref)
    benign = [
        ("seq_nofma", {"preset": "hf_bf16", "expert_fma": "false", "router_fma": "false", "oproj_fma": "false"}),
        ("strided8", {"preset": "hf_bf16", "expert_order": "strided", "expert_block": 8,
                      "router_order": "strided", "router_block": 8, "oproj_order": "strided", "oproj_block": 8}),
        ("pairwise", {"preset": "hf_bf16", "expert_order": "pairwise", "router_order": "pairwise",
                      "oproj_order": "pairwise"}),
    ]
    runs = {n: load_run(run(tmp, n, export, num)) for n, num in benign}
    tol = calibrate([(r, runs["seq_nofma"]), (r, runs["strided8"])], margin=1.0)
    assert all(tol[op] >= 0 for op in ("router_logits", "output"))
    # The held-out benign pair is mostly within tolerance; a planted fp16 expert accumulator is far outside it.
    held = compare(r, runs["pairwise"], tol)
    bug = run(tmp, "cal_bug", export, {"preset": "hf_bf16"},
              {"bug_id": "x", "type": "expert_accum", "params": {"dtype": "fp16"}})
    rep = localize(r, load_run(bug), tol, rerun_template(tmp, "cal_bug", export))
    assert rep["localized_op"] == "expert_gate", rep["verdict"]
    worst_benign = max(held["ops"][op]["max_scaled_ulp"] for op in ("expert_gate", "output"))
    worst_bug = rep["comparison"]["ops"]["expert_gate"]["max_scaled_ulp"]
    assert worst_bug > 4 * worst_benign
