# routediff report

- Reference: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer7-float32/cpu` (torch-cpu)
- Candidate: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer7-float32/cuda` (torch-cuda)
- Tokens: 64 (split: all); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 132 | 502 | 0 | 304 | 6.46e-07 |
| 2 | `resid_sum` |  | 13 | 502 | 0 | 48 | 3.05e-09 |
| 3 | `router_input` |  | 15 | 366 | 0 | 48 | 1.14e-07 |
| 4 | `router_logits` |  | 36 | 122 | 0 | 413 | 9.07e-07 |
| 5 | `router_probs` |  | 64 | 128 | 0 | 43 | 1.25e-06 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 64 | 128 | 0 | 43 | 1.45e-06 |
| 8 | `expert_gate` |  | 86 | 156 | 0 | 199 | 8.18e-07 |
| 9 | `expert_up` |  | 86 | 194 | 0 | 229 | 8.55e-07 |
| 10 | `expert_act` |  | 129 | 439 | 0 | 428 | 1.11e-06 |
| 11 | `expert_down` |  | 139 | 526 | 0 | 958 | 1.21e-06 |
| 12 | `combine_terms` |  | 193 | 432 | 0 | 964 | 1.32e-06 |
| 13 | `output` |  | 83.5 | 339 | 0 | 1.05e+03 | 1.18e-06 |
