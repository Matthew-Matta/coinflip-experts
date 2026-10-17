# routediff report

- Reference: `data/results/simd/layer0-cpp-hf_fp32-scalar` (cpp)
- Candidate: `data/results/simd/layer0-cpp-hf_fp32-simd-accs8` (cpp)
- Tokens: 664 (split: evaluation); tokens whose expert set differs: 0

**Verdict:** localized to o_proj_out (fault touches o_proj_out, router_logits, expert_gate, expert_up, expert_down): patching restores the reference (output_tolerance)

| # | op | flagged | worst scaled ULP | tolerance | tokens over | p99 elem ULP | rel L2 |
|---|---|---|---|---|---|---|---|
| 1 | `o_proj_out` | **yes** | 282 | 0 | 664 | 274 | 6.95e-07 |
| 2 | `resid_sum` | **yes** | 263 | 0 | 664 | 288 | 3.95e-07 |
| 3 | `router_input` | **yes** | 144 | 0 | 664 | 293 | 5.47e-07 |
| 4 | `router_logits` | **yes** | 48 | 0 | 664 | 154 | 7.89e-07 |
| 5 | `router_probs` | **yes** | 48 | 0 | 664 | 39 | 8.45e-07 |
| 6 | `topk_ids` |  | set mismatches: 0 (unexplained 0) | | | | |
| 7 | `topk_weights` | **yes** | 48 | 0 | 664 | 26 | 9.14e-07 |
| 8 | `expert_gate` | **yes** | 78 | 0 | 664 | 331 | 7.86e-07 |
| 9 | `expert_up` | **yes** | 126 | 0 | 664 | 520 | 8.94e-07 |
| 10 | `expert_act` | **yes** | 181 | 0 | 664 | 854 | 1.13e-06 |
| 11 | `expert_down` | **yes** | 352 | 0 | 664 | 865 | 1.08e-06 |
| 12 | `combine_terms` | **yes** | 203 | 0 | 664 | 864 | 1.44e-06 |
| 13 | `output` | **yes** | 117 | 0 | 664 | 846 | 1.21e-06 |

## Patch checks (cumulative)

| patched so far | criterion | restored | routes back | output back | still flagged |
|---|---|---|---|---|---|
| `o_proj_out` | output_tolerance | False | True | False | router_logits, router_probs, topk_weights, expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits` | output_tolerance | False | True | False | expert_gate, expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate` | output_tolerance | False | True | False | expert_up, expert_act, expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up` | output_tolerance | False | True | False | expert_down, combine_terms, output |
| `o_proj_out`, `router_logits`, `expert_gate`, `expert_up`, `expert_down` | output_tolerance | True | True | True | - |
