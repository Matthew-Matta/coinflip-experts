# routediff report

- Reference: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer0-bfloat16/cpu` (torch-cpu)
- Candidate: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer0-bfloat16/cuda` (torch-cuda)
- Tokens: 64 (split: all); tokens whose expert set differs: 0

**Verdict:** localized to router_probs: patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 1.30e-04 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 9.88e-05 |
| 3 | `router_input` |  | 2 | 6 | 0 | 0 | 1.51e-04 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 12 | 1.90e-03 |
| 5 | `router_probs` | **yes** | 1.52e+05 | 1.29e+05 | 3 | 1.35e+05 | 2.85e-03 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 3 | 2 | 1 | 2 | 3.50e-03 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 1 | 4.48e-04 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 1 | 4.97e-04 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 2 | 5.15e-04 |
| 11 | `expert_down` |  | 2 | 9.44 | 0 | 5 | 8.05e-04 |
| 12 | `combine_terms` |  | 5 | 9.19 | 0 | 5 | 3.50e-03 |
| 13 | `output` |  | 7 | 482 | 0 | 17 | 3.23e-03 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `router_probs` | output_tolerance | True | True | True | - |
