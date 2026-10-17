# routediff report

- Reference: `<repo>/data/results/torch-hook-demo-cpu/clean` (torch-cpu)
- Candidate: `<repo>/data/results/torch-hook-demo-cpu/faulty` (torch-cpu)
- Tokens: 64 (split: all); tokens whose expert set differs: 0

**Verdict:** localized to combine_terms: patching restores the reference (output_tolerance)

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
| 12 | `combine_terms` | **yes** | 3.82e+03 | 9.19 | 64 | 3.81e+03 | 1.50e+01 |
| 13 | `output` | **yes** | 3.82e+03 | 482 | 64 | 3.81e+03 | 1.50e+01 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `combine_terms` | output_tolerance | True | True | True | - |
