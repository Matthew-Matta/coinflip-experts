# routediff report

- Reference: `data/dumps/layer0-torch-bfloat16` (torch-cpu)
- Candidate: `data/results/bf16-tolerances/layer0-cpp-bf16-cal-0` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 6

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 4.87e-05 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 2.99e-05 |
| 3 | `router_input` |  | 3 | 6 | 0 | 0 | 4.63e-05 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 0 | 1.37e-04 |
| 5 | `router_probs` |  | 7.52e+04 | 1.29e+05 | 0 | 866 | 1.78e-04 |
| 6 | `topk_ids` |  | set mismatches: 6 (unexplained 6) | | | | |
| 7 | `topk_weights` |  | 1 | 2 | 0 | 0 | 2.48e-04 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 0 | 1.40e-04 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 0 | 1.74e-04 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 0 | 2.28e-04 |
| 11 | `expert_down` |  | 2 | 9.44 | 0 | 1 | 4.06e-04 |
| 12 | `combine_terms` |  | 3 | 9.19 | 0 | 1 | 4.41e-04 |
| 13 | `output` |  | 98.5 | 482 | 0 | 5 | 1.70e-02 |
