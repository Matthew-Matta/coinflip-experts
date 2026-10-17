# bf16 tolerances for routediff

**In one line:** with tolerances learned in bf16, Apple GPU (MPS) and grouped_mm bf16 are **inside the benign
envelope at every op**; but the bf16 output tolerance (**482**) is useless as a detector until ties are decided like
the reference.
- **What I found:** before the router, a bf16 port and PyTorch agree within 1-3 bf16 ULP (tolerances 2-6). Exact
  ties at 8th/9th place, decided differently, inflate the output tolerance; on route-matched tokens it would be
  **10** (48x tighter).
- **Why it matters:** in bf16, a port must break top-k ties exactly like the reference before output-level checks
  mean anything (numerics contract R4).
- **Caveat:** detection power is untested (no planted bugs in bf16), and the benign envelope comes from three C++
  variants only.

**Type:** exploratory (after preregistration)   ·   **Reproduce:** `bash findings/bf16-tolerances/run.sh` (then `matched_only.py`, below)

---

## Details

**Question.** The frozen tolerances of the blinded evaluation (`findings/routediff-eval-real-v2/tolerances.json`) are
calibrated in fp32, so they cannot judge bf16 pairs (`findings/backend-pairs-v1`). What does the same recipe give in
bf16, and how do the bf16 backends look against it?

**Recipe.** The same as the blinded evaluation (`evaluation/run.py:29-31`; margin 2.0 per PREREG amendment 5dc67ef):
- Reference: torch CPU bf16 eager dumps.
- Candidates: three C++ `hf_bf16` variants (default; pairwise order; FMA-free).
- Layers 0 and 7, **calibration split only** (684 tokens per layer, 6 pairs).
- Binary: `port/build/moe_run` sha256 `c30b6f2f…`, the blinded evaluation's binary (built from commit 9dd291d).
- Output: `data/evaluation/tolerances-bf16-exploratory.json`; sha256 in `sha256.txt`.
- Judged afterwards on the **evaluation split** (664 tokens per layer): MPS eager bf16, CPU grouped_mm bf16, and the
  three C++ variants (held-out benign check). There is no patch-forward rerun: torch backends cannot be patched.

Reproduce (needs the local exports and torch dumps under `data/`): build `port/` at 9dd291d into `port/build`, then
`bash findings/bf16-tolerances/run.sh` (writes dumps and reports to `data/results/bf16-tolerances/`; refuses to
overwrite the tolerance file), then `.venv/bin/python findings/bf16-tolerances/matched_only.py --out NEW.json`.

### Tolerances (scaled ULP in each op's logical dtype)
| op | bf16 (this) | fp32 (blinded evaluation) |
|---|---|---|
| o_proj_out / resid_sum / router_input | 2 / 4 / 6 | 502 / 502 / 366 |
| router_logits | 2 | 122 |
| router_probs (fp32 dtype in hf_bf16) | 129,002 | 128 |
| expert_gate / up / act / down | 2 / 2 / 6 / 9.4 | 156 / 194 / 439 / 526 |
| combine_terms | 9.2 | 432 |
| **output** | **482** | 339 |
| topk_unexplained (fraction of tokens) | 0.0175 | 0 |

- Before the selection, a bf16 port and torch agree to within 1-3 bf16 ULP: tolerances of 2-6.
- `router_probs` is stored in fp32 (softmax in fp32), so one bf16 ULP of logit difference shows up as ~65,000 fp32
  ULP of probability, hence 129,002.
- `topk_unexplained` = 1.75% is the share of calibration tokens with an exact tie at 8th/9th place that C++ (lower id
  wins) and torch CPU (`nth_element`) decide differently.
- **The output tolerance (482) is useless as a detector.** routediff compares `output` on every token, including those
  tie tokens, whose outputs differ completely, so they set the tolerance: about 4x the row RMS.
  - Restricted to route-matched tokens, the same recipe gives an output tolerance of **10** (48x tighter;
    `matched_only.json`).
  - **In bf16, a port must decide ties exactly like the reference (numerics contract R4) before output-level
    tolerances mean anything.**

### Verdicts on the evaluation split
| pair | route-set mismatches | all unexplained (ties)? | routediff (bf16 tol) | output worst, route-matched tokens (tol 10) |
|---|---|---|---|---|
| L0 MPS eager bf16 | 6 / 664 | yes | nothing flagged | 5 (0.5x) |
| L0 CPU grouped_mm bf16 | 0 / 664 | - | nothing flagged | 4 (0.4x) |
| L7 MPS eager bf16 | 4 / 664 | yes | nothing flagged | 5 (0.5x) |
| L7 CPU grouped_mm bf16 | 0 / 664 | - | nothing flagged | 5 (0.5x) |
| C++ benign variants (held out, 3 per layer) | 6 / 4 per layer | yes | nothing flagged | 4-5 (0.4-0.5x) |

- MPS bf16 and grouped_mm bf16 are inside the bf16 benign envelope at every op, also under the tighter matched-token
  output tolerance.
- Their route-set mismatches against torch CPU are all exact ties (the same tokens as C++ vs torch CPU;
  `findings/backend-pairs-v1`).
- grouped_mm's different combine (fp32 sum, one rounding) is visible as an output rel L2 of 3.7e-3 / 3.9e-3, but that is
  4-5 scaled bf16 ULP: a bf16 output cannot resolve it. It is a numerics difference, not a flaggable port error, at
  this dtype.
- Caveats:
  - Detection power of these bf16 tolerances is not tested; no planted bugs were run in bf16.
  - The benign envelope comes from three C++ variants only, all with the lower-id tie rule.
  - Descriptive only; not preregistered.
