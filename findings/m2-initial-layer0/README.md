# M2 initial result: layer 0 (preregistered H-port evaluation)

Frozen before any change to the C++ port. This is the official preregistered outcome for layer 0 (option A).
Any later result is post-fix validation and is reported separately.

- **Verdict: FAIL** (thresholds unchanged: normalised margin > 0.0001, p99 element ULP <= 8.0)
- Routes: 0 mismatches among 1346 tokens above the margin cutoff (pass); 0 of 1348 overall
- Failing ops: router_input, router_logits, router_probs, topk_weights, expert_gate, expert_up, expert_act, expert_down, combine_terms, output

| op | p99 element ULP | max element ULP | pass |
|---|---|---|---|
| `o_proj_out` | 0 | 0 | yes |
| `resid_sum` | 0 | 0 | yes |
| `router_input` | 9 | 17 | **no** |
| `router_logits` | 39.29 | 4.326e+04 | **no** |
| `router_probs` | 18 | 67 | **no** |
| `topk_weights` | 14 | 41 | **no** |
| `expert_gate` | 114 | 3.255e+07 | **no** |
| `expert_up` | 191 | 2.076e+07 | **no** |
| `expert_act` | 304 | 2.232e+07 | **no** |
| `expert_down` | 376 | 3.944e+08 | **no** |
| `combine_terms` | 375 | 3.208e+08 | **no** |
| `output` | 354 | 1.091e+08 | **no** |

## Diagnosis (exploratory, after the evaluation)
Patching `router_input` with PyTorch's value (`dumps/layer0-cpp-hf_fp32-patch_router_input`) makes `router_logits`, `expert_gate` and `expert_up` bit-exact, so the first divergence is the RMSNorm reduction order. Remaining differences come from `exp` (softmax, SiLU) and the down projection.

## bf16 (exploratory, not part of H-port)
11 / 1348 route-set mismatches, all unexplained (exact ties at the 8th place; 13 / 1348 tokens have such a tie). `tiebreak=higher` leaves 2, so PyTorch's tie order is not a simple index rule.

Contents here: `configs/`, `hport_layer0_fp32.json`, `provenance.json`. The frozen read-only originals, including
`dumps/` (APFS clones of the original dumps) and `routediff-first-look/`, are in `data/results/m2-initial-layer0/` (local, gitignored).
