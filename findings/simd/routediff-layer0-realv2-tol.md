# routediff report

- Reference: `data/results/simd/layer0-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer0-cpp-hf_fp32-simd` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 282 | 502 | 0 | 280 | 7.00e-07 |
| 2 | `resid_sum` |  | 272 | 502 | 0 | 288 | 3.98e-07 |
| 3 | `router_input` |  | 146 | 366 | 0 | 299 | 5.52e-07 |
| 4 | `router_logits` |  | 50 | 122 | 0 | 167 | 7.94e-07 |
| 5 | `router_probs` |  | 48 | 128 | 0 | 39 | 8.52e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 48 | 128 | 0 | 27 | 9.23e-07 |
| 8 | `expert_gate` |  | 76 | 156 | 0 | 339 | 7.92e-07 |
| 9 | `expert_up` |  | 118 | 194 | 0 | 532 | 9.05e-07 |
| 10 | `expert_act` |  | 182 | 439 | 0 | 869 | 1.14e-06 |
| 11 | `expert_down` |  | 312 | 526 | 0 | 876 | 1.09e-06 |
| 12 | `combine_terms` |  | 199 | 432 | 0 | 876 | 1.44e-06 |
| 13 | `output` |  | 129 | 339 | 0 | 848 | 1.20e-06 |
