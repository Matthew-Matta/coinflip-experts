# routediff report

- Reference: `data/dumps/layer7-torch-bfloat16` (torch-cpu)
- Candidate: `data/results/bf16-tolerances/layer7-cpp-bf16-cal-1` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 4

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 2.03e-05 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 1.27e-06 |
| 3 | `router_input` |  | 1 | 6 | 0 | 0 | 1.69e-05 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 0 | 3.66e-05 |
| 5 | `router_probs` |  | 3.43e+04 | 1.29e+05 | 0 | 6 | 3.12e-05 |
| 6 | `topk_ids` |  | set mismatches: 4 (unexplained 4) | | | | |
| 7 | `topk_weights` |  | 1 | 2 | 0 | 0 | 6.48e-05 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 0 | 6.24e-05 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 0 | 6.93e-05 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 0 | 1.01e-04 |
| 11 | `expert_down` |  | 2 | 9.44 | 0 | 0 | 2.26e-04 |
| 12 | `combine_terms` |  | 3 | 9.19 | 0 | 0 | 2.29e-04 |
| 13 | `output` |  | 465 | 482 | 0 | 1 | 1.55e-02 |
