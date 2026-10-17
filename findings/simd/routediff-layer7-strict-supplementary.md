# routediff report

- Reference: `data/results/simd/layer7-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer7-cpp-hf_fp32-simd` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** localized to o_proj_out (fault touches o_proj_out, router_logits, expert_gate, expert_up, expert_down): patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` | **yes** | 97 | 0 | 664 | 232 | 5.50e-07 |
| 2 | `resid_sum` | **yes** | 37 | 0 | 664 | 120 | 1.41e-08 |
| 3 | `router_input` | **yes** | 44.2 | 0 | 664 | 116 | 1.97e-07 |
| 4 | `router_logits` | **yes** | 50 | 0 | 664 | 84 | 7.80e-07 |
| 5 | `router_probs` | **yes** | 47 | 0 | 664 | 47 | 6.85e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 47 | 0 | 664 | 22 | 6.56e-07 |
| 8 | `expert_gate` | **yes** | 85 | 0 | 664 | 261 | 7.77e-07 |
| 9 | `expert_up` | **yes** | 92 | 0 | 664 | 348 | 8.15e-07 |
| 10 | `expert_act` | **yes** | 408 | 0 | 664 | 608 | 1.14e-06 |
| 11 | `expert_down` | **yes** | 188 | 0 | 664 | 899 | 1.28e-06 |
| 12 | `combine_terms` | **yes** | 204 | 0 | 664 | 900 | 1.40e-06 |
| 13 | `output` | **yes** | 183 | 0 | 664 | 944 | 1.29e-06 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `o_proj_out` | output_tolerance | False | True | False | router_logits, router_probs, topk_weights, expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits` | output_tolerance | False | True | False | expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate` | output_tolerance | False | True | False | expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up` | output_tolerance | False | True | False | expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up`, `expert_down` | output_tolerance | True | True | True | - |
