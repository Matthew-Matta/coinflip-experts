# routediff report

- Reference: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer7-bfloat16/cpu` (torch-cpu)
- Candidate: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer7-bfloat16/cuda` (torch-cuda)
- Tokens: 64 (split: all); tokens whose expert set differs: 1

**Verdict:** localized to router_probs: patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 1.39e-04 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 1.14e-06 |
| 3 | `router_input` |  | 1 | 6 | 0 | 0 | 6.02e-05 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 18.1 | 2.22e-03 |
| 5 | `router_probs` | **yes** | 1.94e+05 | 1.29e+05 | 1 | 1.16e+05 | 2.08e-03 |
| 6 | `topk_ids` |  | set mismatches: 1 (unexplained 1) | | | | |
| 7 | `topk_weights` | **yes** | 3 | 2 | 1 | 1 | 2.60e-03 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 0 | 2.69e-04 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 0 | 2.89e-04 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 1 | 3.88e-04 |
| 11 | `expert_down` |  | 1.5 | 9.44 | 0 | 2 | 8.90e-04 |
| 12 | `combine_terms` |  | 5 | 9.19 | 0 | 3 | 2.71e-03 |
| 13 | `output` |  | 71.5 | 482 | 0 | 42 | 2.92e-02 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `router_probs` | output_tolerance | True | True | True | - |
