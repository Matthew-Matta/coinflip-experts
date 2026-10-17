# routediff report

- Reference: `data/dumps/layer0-torch-bfloat16` (torch-cpu)
- Candidate: `data/results/bf16-tolerances/layer0-cpp-bf16-cal-1` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 6

**Verdict:** no op exceeds its tolerance

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 1 | 2 | 0 | 0 | 2.00e-05 |
| 2 | `resid_sum` |  | 1 | 4 | 0 | 0 | 1.42e-05 |
| 3 | `router_input` |  | 2 | 6 | 0 | 0 | 2.03e-05 |
| 4 | `router_logits` |  | 1 | 2 | 0 | 0 | 4.97e-05 |
| 5 | `router_probs` |  | 3.54e+04 | 1.29e+05 | 0 | 179 | 2.64e-05 |
| 6 | `topk_ids` |  | set mismatches: 6 (unexplained 6) | | | | |
| 7 | `topk_weights` |  | 1 | 2 | 0 | 0 | 6.14e-05 |
| 8 | `expert_gate` |  | 1 | 2 | 0 | 0 | 6.07e-05 |
| 9 | `expert_up` |  | 1 | 2 | 0 | 0 | 8.75e-05 |
| 10 | `expert_act` |  | 3 | 6 | 0 | 0 | 1.13e-04 |
| 11 | `expert_down` |  | 2 | 9.44 | 0 | 0 | 2.12e-04 |
| 12 | `combine_terms` |  | 3 | 9.19 | 0 | 0 | 2.30e-04 |
| 13 | `output` |  | 98.5 | 482 | 0 | 2 | 1.70e-02 |
