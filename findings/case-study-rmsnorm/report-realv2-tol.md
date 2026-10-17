# routediff report

- Reference: `data/dumps/layer0-torch-float32` (torch-cpu)
- Candidate: `data/results/m2-initial-layer0/dumps/layer0-cpp-hf_fp32` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 0 | 502 | 0 | 0 | 0.00e+00 |
| 2 | `resid_sum` |  | 0 | 502 | 0 | 0 | 0.00e+00 |
| 3 | `router_input` |  | 17 | 366 | 0 | 9 | 2.84e-07 |
| 4 | `router_logits` |  | 31 | 122 | 0 | 41 | 3.53e-07 |
| 5 | `router_probs` |  | 41 | 128 | 0 | 19 | 4.31e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 41 | 128 | 0 | 14 | 4.71e-07 |
| 8 | `expert_gate` |  | 40 | 156 | 0 | 116 | 4.13e-07 |
| 9 | `expert_up` |  | 78 | 194 | 0 | 195 | 4.93e-07 |
| 10 | `expert_act` |  | 80 | 439 | 0 | 312 | 7.51e-07 |
| 11 | `expert_down` |  | 131 | 526 | 0 | 379 | 7.67e-07 |
| 12 | `combine_terms` |  | 135 | 432 | 0 | 378 | 1.11e-06 |
| 13 | `output` |  | 97 | 339 | 0 | 363 | 1.03e-06 |
