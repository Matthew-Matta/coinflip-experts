# routediff report

- Reference: `data/results/simd/layer7-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer7-cpp-hf_fp32-simd` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 97 | 502 | 0 | 232 | 5.50e-07 |
| 2 | `resid_sum` |  | 37 | 502 | 0 | 120 | 1.41e-08 |
| 3 | `router_input` |  | 44.2 | 366 | 0 | 116 | 1.97e-07 |
| 4 | `router_logits` |  | 50 | 122 | 0 | 84 | 7.80e-07 |
| 5 | `router_probs` |  | 47 | 128 | 0 | 47 | 6.85e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 47 | 128 | 0 | 22 | 6.56e-07 |
| 8 | `expert_gate` |  | 85 | 156 | 0 | 261 | 7.77e-07 |
| 9 | `expert_up` |  | 92 | 194 | 0 | 348 | 8.15e-07 |
| 10 | `expert_act` |  | 408 | 439 | 0 | 608 | 1.14e-06 |
| 11 | `expert_down` |  | 188 | 526 | 0 | 899 | 1.28e-06 |
| 12 | `combine_terms` |  | 204 | 432 | 0 | 900 | 1.40e-06 |
| 13 | `output` |  | 183 | 339 | 0 | 944 | 1.29e-06 |
