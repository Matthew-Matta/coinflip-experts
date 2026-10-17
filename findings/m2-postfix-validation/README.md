# C++ port vs PyTorch after the RMSNorm fix (layers 0 and 7)

**In one line:** after matching PyTorch's RMSNorm order, the fp32 port is **bit-exact through the router and the
expert gate/up matmuls** and picks the **same experts on 1348/1348** tokens per layer; the strict ULP test still
**fails** after `exp`.
- **What I found:** the remaining 2-5 ULP differences start at PyTorch's vectorised `exp` (softmax, SiLU) and grow to
  80-101 ULP p99 where outputs nearly cancel; port and PyTorch are equally close to fp64 (relative L2 1.06e-6 vs
  1.06e-6 at the expert down projection, layer 0).
- **Why it matters:** the port is as accurate as PyTorch; a bit-exact port would also need PyTorch's `exp`.
- **Caveat:** this is post-fix validation; the preregistered outcome is the FAIL in `findings/m2-initial-layer0`.

**Type:** descriptive (post-fix validation, thresholds unchanged)   ·   **Reproduce:** `python -m routediff.fidelity data/dumps/layer0-torch-float32 data/results/m2-postfix-validation/dumps/layer0-cpp-hf_fp32-postfix --out NEW.json`

---

## Details

**Label: post-fix validation.** The preregistered result is `findings/m2-initial-layer0` (layer 0: FAIL). These runs
come after fixing the RMSNorm reduction order that patching located; thresholds are unchanged.

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

### Fixes applied
1. RMSNorm variance summed in PyTorch's CPU order (4-lane cascade, `aten/src/ATen/native/cpu/SumKernel.cpp`).
2. `rsqrt` as `1/sqrt` with two fp32 roundings.
3. `eps` rounded to fp32 before the add (a first post-fix run without this matched `router_input` on 1275/1348 layer-0 tokens; kept as a record).

### Remaining difference (diagnosed by patching)
Through the router and the expert gate/up projections the port is now bit-exact on both layers. Patching PyTorch's `expert_act` into the C++ run makes `expert_down` bit-exact (0 ULP), so the down-projection gap is inherited from a 2-ULP activation difference amplified by near-cancelling sums. The remaining `router_probs` / `topk_weights` (5 ULP) and `expert_act` (2 ULP) differences come from `exp`: PyTorch uses a vectorised approximation (Sleef), the port a correctly rounded one.

### bf16 (exploratory)
Route-set mismatches: layer 0 11/1348, layer 7 16/1348, all unexplained (exact ties at the 8th place broken differently). bf16 `router_input` is not bit-exact (upstream bf16 matmul differences: `o_proj_out` matches on 512/1348 and 613/1348 tokens); not investigated further.


### Error vs the fp64 oracle (`oracle_comparison.json`, `oracle_comparison.py`)
Relative L2 vs fp64, C++ / PyTorch: expert_down 1.06e-6 / 1.06e-6 (layer 0), 1.24e-6 / 1.24e-6 (layer 7); output
1.18e-6 / 1.18e-6 (layer 0), 1.26e-6 / 1.26e-6 (layer 7). C++ vs PyTorch directly: 1.2-2.1e-7. The oracle dumps are
stored as fp32 (`spec/INTERFACES.md` section 2).
