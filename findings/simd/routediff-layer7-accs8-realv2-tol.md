# routediff report

- Reference: `data/results/simd/layer7-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer7-cpp-hf_fp32-simd-accs8` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 91 | 502 | 0 | 224 | 5.45e-07 |
| 2 | `resid_sum` |  | 37 | 502 | 0 | 116 | 1.40e-08 |
| 3 | `router_input` |  | 41.5 | 366 | 0 | 114 | 1.97e-07 |
| 4 | `router_logits` |  | 52 | 122 | 0 | 79 | 7.76e-07 |
| 5 | `router_probs` |  | 49 | 128 | 0 | 46 | 6.86e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 49 | 128 | 0 | 21 | 6.62e-07 |
| 8 | `expert_gate` |  | 89 | 156 | 0 | 252 | 7.69e-07 |
| 9 | `expert_up` |  | 81 | 194 | 0 | 336 | 8.05e-07 |
| 10 | `expert_act` |  | 361 | 439 | 0 | 588 | 1.13e-06 |
| 11 | `expert_down` |  | 182 | 526 | 0 | 888 | 1.26e-06 |
| 12 | `combine_terms` |  | 209 | 432 | 0 | 888 | 1.39e-06 |
| 13 | `output` |  | 145 | 339 | 0 | 938 | 1.28e-06 |
