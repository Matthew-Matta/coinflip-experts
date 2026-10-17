# routediff report

- Reference: `data/dumps/layer0-torch-bfloat16` (torch-cpu)
- Candidate: `data/dumps/real-backend-pairs-v1/layer0-cpu-grouped_mm-bfloat16` (torch-cpu)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 2 | `resid_sum` |  | 0 | 4 | 0 | 0 | 0.00e+00 |
| 3 | `router_input` |  | 0 | 6 | 0 | 0 | 0.00e+00 |
| 4 | `router_logits` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 5 | `router_probs` |  | 0 | 1.29e+05 | 0 | 0 | 0.00e+00 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 8 | `expert_gate` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 9 | `expert_up` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 10 | `expert_act` |  | 0 | 6 | 0 | 0 | 0.00e+00 |
| 11 | `expert_down` |  | 0 | 9.44 | 0 | 0 | 0.00e+00 |
| 12 | `combine_terms` |  | 0 | 9.19 | 0 | 0 | 0.00e+00 |
| 13 | `output` |  | 4 | 482 | 0 | 13 | 3.71e-03 |
