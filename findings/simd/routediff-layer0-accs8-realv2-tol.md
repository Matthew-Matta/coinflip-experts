# routediff report

- Reference: `data/results/simd/layer0-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer0-cpp-hf_fp32-simd-accs8` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 282 | 502 | 0 | 274 | 6.95e-07 |
| 2 | `resid_sum` |  | 263 | 502 | 0 | 288 | 3.95e-07 |
| 3 | `router_input` |  | 144 | 366 | 0 | 293 | 5.47e-07 |
| 4 | `router_logits` |  | 48 | 122 | 0 | 154 | 7.89e-07 |
| 5 | `router_probs` |  | 48 | 128 | 0 | 39 | 8.45e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 48 | 128 | 0 | 26 | 9.14e-07 |
| 8 | `expert_gate` |  | 78 | 156 | 0 | 331 | 7.86e-07 |
| 9 | `expert_up` |  | 126 | 194 | 0 | 520 | 8.94e-07 |
| 10 | `expert_act` |  | 181 | 439 | 0 | 854 | 1.13e-06 |
| 11 | `expert_down` |  | 352 | 526 | 0 | 865 | 1.08e-06 |
| 12 | `combine_terms` |  | 203 | 432 | 0 | 864 | 1.44e-06 |
| 13 | `output` |  | 117 | 339 | 0 | 846 | 1.21e-06 |
