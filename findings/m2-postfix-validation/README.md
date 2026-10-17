# M2 post-fix validation (layers 0 and 7)

**Label: post-fix validation.** The preregistered result is `findings/m2-initial-layer0` (layer 0: FAIL). These runs come after fixing the RMSNorm reduction order that routediff-style patching located; thresholds are unchanged.

- Layer 0 post-fix: **FAIL**; fp32 routes 0/1348 mismatches; failing ops: expert_down, combine_terms, output
- Layer 7 post-fix: **FAIL**; fp32 routes 0/1348 mismatches; failing ops: expert_down, combine_terms, output

| op | L0 initial | L0 post-fix | L7 post-fix |
|---|---|---|---|
| `o_proj_out` | 0 | 0 | 0 |
| `resid_sum` | 0 | 0 | 0 |
| `router_input` | 9 (fail) | 0 | 0 |
| `router_logits` | 39.3 (fail) | 0 | 0 |
| `router_probs` | 18 (fail) | 5 | 5 |
| `topk_weights` | 14 (fail) | 5 | 5 |
| `expert_gate` | 114 (fail) | 0 | 0 |
| `expert_up` | 191 (fail) | 0 | 0 |
| `expert_act` | 304 (fail) | 2 | 2 |
| `expert_down` | 376 (fail) | 80 (fail) | 84 (fail) |
| `combine_terms` | 375 (fail) | 80 (fail) | 85 (fail) |
| `output` | 354 (fail) | 82 (fail) | 101 (fail) |

p99 element ULP, fp32, bound 8.

## Fixes applied
1. RMSNorm variance summed in PyTorch's CPU order (4-lane cascade, `aten/src/ATen/native/cpu/SumKernel.cpp`).
2. `rsqrt` as `1/sqrt` with two fp32 roundings.
3. `eps` rounded to fp32 before the add (a first post-fix run without this matched `router_input` on 1275/1348 layer-0 tokens; kept as a record).

## Remaining difference (diagnosed by patching)
Through the router and the expert gate/up projections the port is now bit-exact on both layers. Patching PyTorch's `expert_act` into the C++ run makes `expert_down` bit-exact (0 ULP), so the down-projection gap is inherited from a 2-ULP activation difference amplified by near-cancelling sums. The remaining `router_probs` / `topk_weights` (5 ULP) and `expert_act` (2 ULP) differences come from `exp`: PyTorch uses a vectorised approximation (Sleef), the port a correctly rounded one.

## bf16 (exploratory)
Route-set mismatches: layer 0 11/1348, layer 7 16/1348, all unexplained (exact ties at the 8th place broken differently). bf16 `router_input` is not bit-exact (upstream bf16 matmul differences: `o_proj_out` matches on 512/1348 and 613/1348 tokens); not investigated further.
