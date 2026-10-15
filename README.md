# Coin-Flip Experts

**When you port a Mixture-of-Experts model to new hardware, which operation makes it send a token to a different expert?**

An MoE router picks the top-k experts for every token. When the k-th and (k+1)-th router scores are nearly tied, the choice is decided by rounding: accumulator width, reduction order, the dtype of the gate output, the order experts are combined in, how a collective sums partial results across chips. A port can therefore match the reference to within a few ULP almost everywhere and still route some tokens differently, which changes which weights run and what the model outputs.

This repo builds the tools to find and measure that:

- **`routediff`**: compares an MoE layer across two backends op by op from per-op tensor dumps, names the first operation where routing diverges, and confirms the cause by patching that operation and checking that routing is restored.
- **C++ MoE block port**: a reference implementation of a real model's MoE block (OLMoE) in which every numerics choice a chip has to make is an explicit setting, checked against PyTorch and an fp64 oracle, with a SIMD path whose speedup and reduction-order effect are both measured.
- **Collectives**: how the dtype and topology (ring / tree) of a cross-chip reduction, and the number of chips (2-64, emulated), change route flips, set against a communication cost model (bytes on the wire, steps).
- **A numerics contract**: what an accelerator port must guarantee to keep route flips within a budget, each requirement backed by a measurement from this repo.
- **One SGLang measurement**: how often prefill routing in SGLang changes with batch size alone, measured against a run-to-run noise floor.

> **Status: in progress (Oct 2026).** No results are reported yet. Hypotheses, baselines, thresholds and the analysis plan are fixed in advance in [`PREREG.md`](PREREG.md).

## Components and status
| Component | Where | Status |
|---|---|---|
| C++ MoE block: router, top-k, SwiGLU experts, combine; fp16/bf16/fp32/fp64 emulation with round-to-nearest-even; sequential / pairwise / blocked reduction | `port/` | Skeleton + tests |
| Real OLMoE block export and op-by-op C++ vs PyTorch comparison | `port/`, `routediff/` | Next |
| `routediff` CLI, patch-forward check, planted-bug suite | `routediff/` | Planned |
| `routediff` on real backend pairs (PyTorch CPU vs Apple GPU; eager vs grouped experts) | `routediff/` | Planned |
| Collective emulation + communication cost model | `port/`, `analysis/` | Planned |
| SIMD (NEON) path with measured speedup | `port/` | Planned |
| Numerics contract for accelerator ports | `spec/` | Planned |
| SGLang prefill census (OLMoE, batch {1, 8, 32}, noise floor) | `modal/`, `census/` | Planned |
| Run provenance (git sha, GPU, driver, library versions, config hash) on every result | `coinflip/provenance.py` | Done + tests |

## Method in one paragraph
Every comparison is paired: two configurations see exactly the same input tokens, and a "flip" is a difference in the set of top-k experts at the same (token, layer). Effects are reported as excess over a noise floor (the same configuration run twice), with 95% confidence intervals from a bootstrap over prompts. The SGLang measurement covers **prefill only**; decode with a KV cache is a different code path and is not claimed. `routediff` is scored on bugs injected by a seeded script it has no access to, and only on operations the C++ block actually implements; its false-positive rate is measured on benign pairs (configurations that differ only in permitted numerics).

## Quick start
```bash
pip install -e ".[dev]" && pytest                                                   # Python utilities
cmake -S port -B port/build && cmake --build port/build && ./port/build/test_moe    # C++ port
bash tools/sglang_audit.sh    # lists where SGLang's numerics can depend on batch size
```

## Known limitations
- Cross-chip results for more than 2 chips are emulation, not measurements on real interconnects.
- The C++ port covers the MoE block only (no attention, KV cache or RoPE), so `routediff` is validated only on bugs in those operations.
- The SGLang measurement is prefill only, on one model and one GPU type.
- The SGLang kernel wheels tested (`sglang-kernel` 0.4.6.post1 and 0.4.7, `sgl-kernel` 0.3.21) contain no `sm_75` code, so they do not run on T4 GPUs; GPU runs use an L4 (`sm_89`).

## Related work
Route flips from accumulator precision and summation order: arXiv 2607.28097. Route flips under KV-cache quantization, with activation patching: arXiv 2608.11212. Routing replay between rollout and training: R3 (arXiv 2510.11370) and Miles (arXiv 2609.08368). Expert-selection near-ties in production serving: Fireworks AI blog (2026-09-30). Batch-invariant kernels: Thinking Machines; SGLang `--enable-deterministic-inference`. First-divergence tensor comparison across backends: msprobe, Polygraphy, DeepView.
