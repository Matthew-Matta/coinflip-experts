# routediff report

- Reference: `<repo>/data/results/torch-hook-demo-mps-router-postlayout/clean` (torch-mps)
- Candidate: `<repo>/data/results/torch-hook-demo-mps-router-postlayout/faulty` (torch-mps)
- Tokens: 64 (split: all); tokens whose expert set differs: 64

**Verdict:** localized to router_logits: patching restores the reference (route_sets)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 2 | `resid_sum` |  | 0 | 4 | 0 | 0 | 0.00e+00 |
| 3 | `router_input` |  | 0 | 6 | 0 | 0 | 0.00e+00 |
| 4 | `router_logits` | **yes** | 510 | 2 | 64 | 508 | 2.00e+00 |
| 5 | `router_probs` | **yes** | 3.81e+08 | 1.29e+05 | 64 | 5.15e+09 | 1.89e+00 |
| 6 | `topk_ids` |  | set mismatches: 64 (unexplained 0) | | | | |
| 7 | `topk_weights` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 8 | `expert_gate` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 9 | `expert_up` |  | 0 | 2 | 0 | 0 | 0.00e+00 |
| 10 | `expert_act` |  | 0 | 6 | 0 | 0 | 0.00e+00 |
| 11 | `expert_down` |  | 0 | 9.44 | 0 | 0 | 0.00e+00 |
| 12 | `combine_terms` |  | 0 | 9.19 | 0 | 0 | 0.00e+00 |
| 13 | `output` | **yes** | 1.1e+04 | 482 | 33 | 3.8e+04 | 3.88e+00 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `router_logits` | route_sets | True | True | True | - |
