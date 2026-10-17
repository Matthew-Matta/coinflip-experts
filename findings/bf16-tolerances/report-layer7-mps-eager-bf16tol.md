# routediff report

- Reference: `data/dumps/layer7-torch-bfloat16` (torch-cpu)
- Candidate: `data/dumps/real-backend-pairs-v1/layer7-mps-eager-bfloat16` (torch-mps)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 4

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 4.02e-05 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 2.18e-06 |
| 3 | `router_input` |  | 3 | 6 | 0 | 0 | 3.59e-05 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 0 | 7.10e-05 |
| 5 | `router_probs` |  | 3.93e+04 | 1.29e+05 | 0 | 784 | 6.33e-05 |
| 6 | `topk_ids` |  | set mismatches: 4 (unexplained 4) | | | | |
| 7 | `topk_weights` |  | 1 | 2 | 0 | 0 | 1.59e-04 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 0 | 1.28e-04 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 0 | 1.39e-04 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 0 | 1.88e-04 |
| 11 | `expert_down` |  | 2 | 9.44 | 0 | 1 | 4.15e-04 |
| 12 | `combine_terms` |  | 3 | 9.19 | 0 | 1 | 4.44e-04 |
| 13 | `output` |  | 465 | 482 | 0 | 3 | 1.55e-02 |
