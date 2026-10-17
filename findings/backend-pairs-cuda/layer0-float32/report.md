# routediff report

- Reference: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer0-float32/cpu` (torch-cpu)
- Candidate: `/weights/numerics-followup-results/84cdd47f1494/cuda-pairs/layer0-float32/cuda` (torch-cuda)
- Tokens: 64 (split: all); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 181 | 502 | 0 | 304 | 7.79e-07 |
| 2 | `resid_sum` |  | 234 | 502 | 0 | 320 | 5.18e-07 |
| 3 | `router_input` |  | 148 | 366 | 0 | 324 | 6.98e-07 |
| 4 | `router_logits` |  | 39 | 122 | 0 | 502 | 8.41e-07 |
| 5 | `router_probs` |  | 54 | 128 | 0 | 44 | 1.31e-06 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 44 | 128 | 0 | 38 | 1.38e-06 |
| 8 | `expert_gate` |  | 62 | 156 | 0 | 468 | 8.91e-07 |
| 9 | `expert_up` |  | 79 | 194 | 0 | 638 | 9.73e-07 |
| 10 | `expert_act` |  | 138 | 439 | 0 | 1.1e+03 | 1.13e-06 |
| 11 | `expert_down` |  | 146 | 526 | 0 | 911 | 1.01e-06 |
| 12 | `combine_terms` |  | 203 | 432 | 0 | 914 | 1.54e-06 |
| 13 | `output` |  | 87 | 339 | 0 | 976 | 1.23e-06 |
