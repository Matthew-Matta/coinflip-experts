# routediff report

- Reference: `data/results/simd/layer7-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer7-cpp-hf_fp32-simd-accs8` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** localized to o_proj_out (fault touches o_proj_out, router_logits, expert_gate, expert_up, expert_down): patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` | **yes** | 91 | 0 | 664 | 224 | 5.45e-07 |
| 2 | `resid_sum` | **yes** | 37 | 0 | 664 | 116 | 1.40e-08 |
| 3 | `router_input` | **yes** | 41.5 | 0 | 664 | 114 | 1.97e-07 |
| 4 | `router_logits` | **yes** | 52 | 0 | 664 | 79 | 7.76e-07 |
| 5 | `router_probs` | **yes** | 49 | 0 | 664 | 46 | 6.86e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 49 | 0 | 664 | 21 | 6.62e-07 |
| 8 | `expert_gate` | **yes** | 89 | 0 | 664 | 252 | 7.69e-07 |
| 9 | `expert_up` | **yes** | 81 | 0 | 664 | 336 | 8.05e-07 |
| 10 | `expert_act` | **yes** | 361 | 0 | 664 | 588 | 1.13e-06 |
| 11 | `expert_down` | **yes** | 182 | 0 | 664 | 888 | 1.26e-06 |
| 12 | `combine_terms` | **yes** | 209 | 0 | 664 | 888 | 1.39e-06 |
| 13 | `output` | **yes** | 145 | 0 | 664 | 938 | 1.28e-06 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `o_proj_out` | output_tolerance | False | True | False | router_logits, router_probs, topk_weights, expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits` | output_tolerance | False | True | False | expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate` | output_tolerance | False | True | False | expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up` | output_tolerance | False | True | False | expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up`, `expert_down` | output_tolerance | True | True | True | - |
