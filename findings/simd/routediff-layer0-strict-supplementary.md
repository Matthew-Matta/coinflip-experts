# routediff report

- Reference: `data/results/simd/layer0-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer0-cpp-hf_fp32-simd` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** localized to o_proj_out (fault touches o_proj_out, router_logits, expert_gate, expert_up, expert_down): patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` | **yes** | 282 | 0 | 664 | 280 | 7.00e-07 |
| 2 | `resid_sum` | **yes** | 272 | 0 | 664 | 288 | 3.98e-07 |
| 3 | `router_input` | **yes** | 146 | 0 | 664 | 299 | 5.52e-07 |
| 4 | `router_logits` | **yes** | 50 | 0 | 664 | 167 | 7.94e-07 |
| 5 | `router_probs` | **yes** | 48 | 0 | 664 | 39 | 8.52e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 48 | 0 | 664 | 27 | 9.23e-07 |
| 8 | `expert_gate` | **yes** | 76 | 0 | 664 | 339 | 7.92e-07 |
| 9 | `expert_up` | **yes** | 118 | 0 | 664 | 532 | 9.05e-07 |
| 10 | `expert_act` | **yes** | 182 | 0 | 664 | 869 | 1.14e-06 |
| 11 | `expert_down` | **yes** | 312 | 0 | 664 | 876 | 1.09e-06 |
| 12 | `combine_terms` | **yes** | 199 | 0 | 664 | 876 | 1.44e-06 |
| 13 | `output` | **yes** | 129 | 0 | 664 | 848 | 1.20e-06 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `o_proj_out` | output_tolerance | False | True | False | router_logits, router_probs, topk_weights, expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits` | output_tolerance | False | True | False | expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate` | output_tolerance | False | True | False | expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up` | output_tolerance | False | True | False | expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up`, `expert_down` | output_tolerance | True | True | True | - |
