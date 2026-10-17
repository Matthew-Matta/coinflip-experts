# Real backend pairs: descriptive analysis (items 2 and 3)

Descriptive, no planted ground truth.

## Item 2: grouped_mm vs eager (bf16), each against the fp64 oracle
Tokens where both backends chose the same expert set as the oracle (combine/expert numerics isolated from routing). Paired per-token relative L2 error; 95% CI by cluster bootstrap over prompts (10,000 resamples, seed 20261016).

| layer | split | tokens | eager rel err | grouped_mm rel err | grouped closer on | mean diff (grouped - eager) [95% CI] |
|---|---|---|---|---|---|---|
| 0 | all | 1338 | 5.718e-03 | 4.752e-03 | 96.9% | -9.66e-04 [-1.01e-03, -9.27e-04] |
| 0 | evaluation | 659 | 5.660e-03 | 4.748e-03 | 96.7% | -9.13e-04 [-9.64e-04, -8.67e-04] |
| 7 | all | 1331 | 6.434e-03 | 5.623e-03 | 99.5% | -8.12e-04 [-8.35e-04, -7.88e-04] |
| 7 | evaluation | 659 | 6.427e-03 | 5.601e-03 | 99.4% | -8.26e-04 [-8.61e-04, -7.91e-04] |

Router logits and expert sets are identical between the two backends; the difference is entirely after routing.
**Mechanism (verified, 100% of tokens on both layers):** `grouped_mm` sums the 8 weighted expert outputs in fp32 (`.sum(dim=1)`, slot order) and rounds to bf16 once (transformers 5.12.1 `integrations/moe.py:475-481`); eager `index_add_`s into a bf16 buffer, rounding after each expert in ascending id (`models/olmoe/modeling_olmoe.py:336`). Recomputing each backend's output from its own `combine_terms` with those rules reproduces it bit-exactly.

## Item 3: bf16 ties at the 8th/9th expert
- Layer 0: 13 tie tokens, all with bit-identical bf16 logits in torch CPU, torch MPS and C++. Same chosen set: C++ vs MPS 13/13, torch CPU vs MPS 2/13, torch CPU vs C++ 2/13.
- Layer 7: 25 tie tokens, all with bit-identical bf16 logits in torch CPU, torch MPS and C++. Same chosen set: C++ vs MPS 25/25, torch CPU vs MPS 9/25, torch CPU vs C++ 9/25.
- C++ (`tiebreak=lower`) and MPS agree on every tie (38/38). Torch CPU differs on 27/38.
- **Torch CPU's tie behaviour is reproduced exactly** (1348/1348 tokens on both layers, set and order) by emulating `aten/src/ATen/native/cpu/SortingKernel.cpp` topk: (value, index) pairs, value-only comparator, `std::nth_element(k-1)` then `std::sort` of the first k-1 (since k*64 > n), with Apple clang's libc++ (`torch_topk_emulation.cpp`). Ties are therefore decided by the standard library's selection algorithm, not by expert id; no simple id rule exists.
- Untested hypothesis: Linux builds (libstdc++) may break the same ties differently.
- fp32 probabilities are never bit-identical across the three backends (different `exp`), so ties are identified on the bf16 logits.

## Inputs
- oracle: `oracle-layer{0,7}` from `port/build/moe_run` (sha256 c30b6f2faee45cb2), preset oracle
- torch CPU eager: `data/dumps/layer{0,7}-torch-bfloat16`; grouped_mm and MPS: `data/dumps/real-backend-pairs-v1/`; C++: `data/results/m2-postfix-validation/dumps/layer{0,7}-cpp-hf_bf16-postfix`
- git: 9dd291d

## Item 1: routediff on each pair (frozen real-v2 tolerances, evaluation split, no rerun)
Reference: PyTorch CPU eager in the same dtype. Tolerances `data/evaluation/real-v2/tolerances.json` (sha256 1944f35dfec6c5e3ff108f482cc47c8effb7a20e8c1d820d2d5c92d4cfd886ff). These are PyTorch backends with no patch support, so there is no patch-forward check: flagged ops only, not localisation.

| pair | route-set mismatches | unexplained | flagged ops | output rel L2 |
|---|---|---|---|---|
| layer0-cpu-grouped_mm-bfloat16 | 0/664 | 0 | none | 3.71e-03 |
| layer0-cpu-grouped_mm-float32 | 0/664 | 0 | none | 6.76e-08 |
| layer0-mps-eager-bfloat16 | 6/664 | 6 | router_probs, topk_ids | 1.70e-02 |
| layer0-mps-eager-float32 | 0/664 | 0 | none | 2.70e-07 |
| layer7-cpu-grouped_mm-bfloat16 | 0/664 | 0 | none | 3.94e-03 |
| layer7-cpu-grouped_mm-float32 | 0/664 | 0 | none | 7.35e-08 |
| layer7-mps-eager-bfloat16 | 4/664 | 4 | router_probs, topk_ids, output | 1.55e-02 |
| layer7-mps-eager-float32 | 0/664 | 0 | none | 3.14e-07 |

- **fp32 (interpretable):** MPS and grouped_mm are within the benign envelope on both layers: nothing flagged, no route mismatches.
- **bf16, routing:** MPS changes the expert set on 6 (L0) and 4 (L7) evaluation tokens, all unexplained (the tie signature of item 3). `router_probs` is the first flagged op; probabilities are fp32 even in bf16 runs, so the fp32-calibrated tolerance applies there, and they exceed the benign envelope. This is not attributed to MPS's softmax: MPS bf16 already differs from the first op (`o_proj_out`, item 3 inputs), upstream bf16 differences are not flagged only because bf16 ops are measured in bf16 ULP, and there is no patch-forward check on torch backends.
- **bf16, value ops: not interpretable with these tolerances.** They are counts of ULP calibrated on fp32 pairs; on a bf16 op the same count means bf16 ULP, about 65,000x looser. That is why grouped_mm bf16, whose combine differs measurably (item 2), shows no flags. A bf16 verdict would need tolerances calibrated on bf16 benign pairs, which the preregistered (fp32) design does not include.
