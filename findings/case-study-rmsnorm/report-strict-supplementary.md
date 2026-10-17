# routediff report

- Reference: `data/dumps/layer0-torch-float32` (torch-cpu)
- Candidate: `data/results/m2-initial-layer0/dumps/layer0-cpp-hf_fp32` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** localized to router_input (fault touches router_input, router_probs, expert_act): patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` |  | 0 | 0 | 0 | 0 | 0.00e+00 |
| 2 | `resid_sum` |  | 0 | 0 | 0 | 0 | 0.00e+00 |
| 3 | `router_input` | **yes** | 17 | 0 | 581 | 9 | 2.84e-07 |
| 4 | `router_logits` | **yes** | 31 | 0 | 581 | 41 | 3.53e-07 |
| 5 | `router_probs` | **yes** | 41 | 0 | 664 | 19 | 4.31e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 41 | 0 | 659 | 14 | 4.71e-07 |
| 8 | `expert_gate` | **yes** | 40 | 0 | 581 | 116 | 4.13e-07 |
| 9 | `expert_up` | **yes** | 78 | 0 | 581 | 195 | 4.93e-07 |
| 10 | `expert_act` | **yes** | 80 | 0 | 664 | 312 | 7.51e-07 |
| 11 | `expert_down` | **yes** | 131 | 0 | 664 | 379 | 7.67e-07 |
| 12 | `combine_terms` | **yes** | 135 | 0 | 664 | 378 | 1.11e-06 |
| 13 | `output` | **yes** | 97 | 0 | 664 | 363 | 1.03e-06 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `router_input` | output_tolerance | False | True | False | router_probs, topk_weights, expert_act, expert_down, combine_terms, output |
| `router_input`, `router_probs` | output_tolerance | False | True | False | expert_act, expert_down, combine_terms, output |
| `router_input`, `router_probs`, `expert_act` | output_tolerance | True | True | True | - |
