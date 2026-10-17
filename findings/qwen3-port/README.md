# Porting a second model: Qwen3-30B-A3B, layer 0

**In one line:** pointing the C++ port at a second MoE model took **about 15 minutes**, and in fp32 it picks the
same experts as PyTorch on **1361 / 1361** tokens.
- **What I found:** bit-exact with PyTorch through the router logits; two real changes were needed (a rectangular
  `o_proj` and the order of the top-k renormalisation sum). In bf16, **9.1%** of tokens tie exactly at the top-8
  cutoff (OLMoE: 1.0-1.9%).
- **Why it matters:** the porting tooling carries over to a new model quickly, and the tie-break rule matters
  5-9x more on this model.
- **Caveat:** one layer, one model; the strict 8-ULP element criterion still fails after `exp` (59-60 ULP p99).

**Type:** exploratory (after preregistration)   ·   **Reproduce:** `port/build/moe_run --export data/export/qwen3-30b-a3b-layer0/layer0 --numerics findings/qwen3-port/numerics-hf_fp32.json --dump NEW_DIR` (full steps below)

---

## Details

**Question.** How much work does it take to point the C++ port at a different MoE model, and does it match PyTorch?

- **Reference:** the export `data/export/qwen3-30b-a3b-layer0`: HF revision `ad44e777…`, transformers 5.12.1,
  eager experts, 1361 tokens. Torch CPU fp32 and bf16 dumps; mirror == module.
- **Candidate:** C++ port, `port/build-qwen/moe_run` sha256 `332454139f3a874ad3150279d593038cac049f01cd20a46dd6eb707a21fc22e6`
  (built from the working tree at the time; the same port changes were committed in 43ee582).
- **fp64 oracle:** the same binary, `oracle` preset.

### What had to change (read from the source first)
transformers 5.12.1 `models/qwen3_moe/modeling_qwen3_moe.py`:
- Router (`:263-272`) and eager experts (`:227-251`) are line for line the same computation as OLMoE's
  (`models/olmoe/modeling_olmoe.py`). The same goes for the decoder residual structure (`:324-354`) and RMSNorm
  (`:299-304`, eps from config: 1e-6 vs 1e-5).
- Differences that matter for the block:
  1. `norm_topk_prob = True`: `router_top_value /= router_top_value.sum(dim=-1, keepdim=True)` in fp32 (`:268-269`),
     before the cast to the model dtype (`:270`). The port already had the switch. **The order of the 8-term sum was
     new.** Given torch's own `router_probs` (patched in), the port's renormalised weights match torch bit-exactly on:
     - 727 / 1361 tokens with a sequential sum;
     - 895 with pairwise;
     - **1361 with the torch CPU SumKernel cascade (4 lanes)**, the same kernel already emulated for RMSNorm
       (`aten/src/ATen/native/cpu/SumKernel.cpp`). That is now the default (`Numerics::renorm_order`), and it does
       nothing when `norm_topk_prob` is false, so OLMoE is unaffected.
  2. **Dimensions from the config:** 128 experts, top-8, hidden 2048, `moe_intermediate_size` 768, and a rectangular
     `o_proj`. The input is 32 heads x 128 = 4096 (`o_proj_input_size` in the export config). The port previously
     assumed a square `o_proj`; it now reads the width from the config (`BlockConfig::oproj_in`).
- **Code change:** `port/include/block.hpp`, `port/src/block.cpp`, `port/tools/moe_run.cpp` (rectangular `o_proj`,
  `renorm_order`/`renorm_block` numerics keys).
- **Tests** (`port/tests/test_moe.cpp`):
  - The null-patch and every-bug-first-shows-at-its-op tests now also run on a Qwen-shaped block (rectangular
    `o_proj`, 32 experts, top-8, renormalised).
  - A new test checks that renormalised weights sum to 1.
  - All pass, as before for the OLMoE-shaped block.

### Timeline (wall clock, 2026-10-17, laptop shared with a background sweep)
| time | step |
|---|---|
| 00:46:38 | export found; start reading `modeling_qwen3_moe.py` |
| ~00:50 | rectangular `o_proj` + Qwen-shaped tests written, tests pass |
| ~00:57 | first C++ runs: routes 1361/1361, `router_input`/`router_logits` bit-exact; renorm order identified by patching torch's `router_probs` (1361/1361 with the cascade) |
| 00:59:04 | full fp32 / bf16 / oracle runs on the real layer |
| 01:01:13 | H-port-style report |
| 01:02:13 | patch-forward check of the remaining fp32 difference |

**About 15 minutes from export to a validated fp32 match,** because the numerics knobs, op dumps, routediff
comparison and patching already existed. Most of the wall time is moe_run on 1361 tokens (40-200 s per run under load).

### fp32: C++ vs torch CPU (H-port criteria from PREREG section 1, applied descriptively)
| op | bit-exact tokens | p99 element ULP | rel L2 C++ vs torch | rel L2 vs fp64 oracle (torch / C++) |
|---|---|---|---|---|
| o_proj_out, resid_sum, router_input, router_logits | 1361 / 1361 | 0 | 0 | 8.3e-7 / same |
| router_probs | 0 | 6 | 1.8e-7 | 2.33e-6 / 2.33e-6 |
| topk_ids (sets) | 1361 / 1361 match | | | 0 / 0 flips vs oracle |
| topk_weights | 67 | 2 | 7.5e-8 | 1.74e-6 / 1.74e-6 |
| expert_gate, expert_up | 1361 / 1361 | 0 | 0 | 8.8e-7 / 9.2e-7 |
| expert_act | 0 | 2 | 5.1e-8 | 1.25e-6 / 1.25e-6 |
| expert_down | 0 | 60 | 9.1e-8 | 1.14e-6 / 1.14e-6 |
| combine_terms | 0 | 59 | 1.2e-7 | 1.97e-6 / 1.97e-6 |
| output | 0 | 60 | 1.1e-7 | 1.34e-6 / 1.34e-6 |

- **Routes:** 0 mismatches on all 1361 tokens (1356 above the preregistered margin cutoff); 0 flips vs the oracle for
  both torch and C++.
- **Element-ULP criterion (p99 <= 8):** fails at `expert_down`, `combine_terms` and `output` (59-60 ULP). This is
  the same pattern as OLMoE after its fix (`findings/m2-postfix-validation`).
- **Patch-forward:** patching torch's `expert_act` into the C++ run makes `expert_down` bit-exact on 1361/1361 tokens.
  The remaining differences all start at torch's vectorised `exp` (softmax: `router_probs`; SiLU: `expert_act`), at
  2-6 ULP, and are amplified where outputs nearly cancel.
- C++ and torch are equally close to the fp64 oracle at every op (relative L2 within 0.5%). The port is as accurate as
  PyTorch; it just isn't bit-identical after `exp`.

### bf16 (exploratory)
- **Route sets:** 65 / 1361 tokens differ from torch CPU, and **all 65 are exact ties** at 8th/9th place in the bf16
  router logits.
- **Qwen3 ties far more often than OLMoE:** 124 / 1361 tokens (9.1%) have an exact 8th/9th tie, vs 13 and 25 of 1348
  (1.0% / 1.9%) on OLMoE layers 0 and 7. That is 128 experts and a top-8 cutoff deeper into a flatter distribution.
  The tie-break rule (numerics contract R4) matters roughly 5-9x more for this model.
- **bf16 `o_proj`:**
  - bit-exact on 583 / 1361 tokens (99.97% of elements). OLMoE's square `o_proj` is no different: 512 / 1348 and
    613 / 1348 tokens bit-exact on layers 0 and 7 (`data/results/m2-postfix-validation/dumps/*-cpp-hf_bf16-postfix`
    vs `data/dumps/layer{0,7}-torch-bfloat16`). So torch's bf16 CPU GEMM accumulates in an order the port does not
    reproduce, for both models; it is not specific to this shape.
  - On 128 tokens: pairwise 100/128 tokens (99.99% of elements), strided-16 95, blocked-256 86, sequential 67.
  - Not identified further (time box). These are single bf16 rounding-boundary cases.
- Output relative L2 vs oracle: torch 1.307e-2, C++ 1.312e-2.

### Files
In this folder: `numerics-*.json`, `h_port_fp32.json`, `per_op_fp32.json`, `renorm_order_identification.json`. The
dump directories named below are in `data/results/qwen3-port/` (local, gitignored).
- `numerics-*.json`, `cpp-hf_fp32`, `cpp-hf_bf16`, `cpp-oracle`: full 13-op dumps.
- `h_port_fp32.json`: `routediff.fidelity` output.
- `per_op_fp32.json`: per-op table.
- `cpp-fp32-renorm-*-patch_router_probs`, `renorm_order_identification.json`: renorm-order identification (regenerated at 01:08 with the final binary after an accidental delete; same counts).
- `cpp-hf_fp32-patch_expert_act`: patch-forward check.

### Reproduce
Build `port/` at 43ee582 or later (`make test-cpp`), then, with the export from `reference/export_qwen.py`:
```sh
for n in hf_fp32 hf_bf16 oracle; do
  port/build/moe_run --export data/export/qwen3-30b-a3b-layer0/layer0 --numerics findings/qwen3-port/numerics-$n.json \
      --dump data/results/qwen3-port/cpp-$n --threads 4
done
python -m routediff.fidelity data/export/qwen3-30b-a3b-layer0/torch-float32 data/results/qwen3-port/cpp-hf_fp32 \
    --out h_port_fp32.json
```
