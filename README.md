# Coin-Flip Experts

> Every number below is copied from the `findings/` file named next to it. Lines marked *exploratory* were run
> after preregistration.

A Mixture-of-Experts (MoE) language model sends every word to a few specialist sub-networks, called experts.
A small "router" scores all 64 experts and picks the top 8. Sometimes expert #8 and expert #9 score almost exactly
the same. Then tiny rounding differences decide which one runs: which chip does the math, the order numbers are
added in, even how many other requests share the batch. For that word, the choice is a coin flip, and a different
expert means a different computation. This project measures how often that happens, finds which step of the math
caused it, and turns the answers into requirements that a chip running the model has to meet.

## Headline
- **Batch size alone changes which experts run for 7.25% of word-layers** in SGLang (OLMoE, prefill, batch 8 vs 1; 0% when the same run repeats).
- **Our debugger, routediff, found and pinpointed 44/44 planted porting bugs with 0/32 false alarms** (blinded, preregistered).
- **Recovered real NCCL's 4-GPU summation order from its outputs alone, 100% bit-exact** (Tree: ranks 2, 3, 1, 0; Ring: start rank rotates per block; 4x L4, post hoc).
- **routediff runs on 4 backends, and a second model was ported in ~15 minutes** (C++ port, PyTorch CPU, Apple GPU, NVIDIA L4; Qwen3-30B-A3B layer: 1361/1361 routes match).
- **The router's adder needs at least 18 bits of precision, its input at least 19** (mantissa bits for 0 route flips; bf16's 7 flips 13%; exploratory).

What a chip must guarantee, each line backed by one of these measurements: [`spec/NUMERICS_CONTRACT.md`](spec/NUMERICS_CONTRACT.md)
(builds on the compatibility contract proposed in arXiv 2607.28097).

![Share of expert choices that change with batch size, per layer](findings/census-olmoe-prefill/per_layer_flips.png)

*In SGLang, running the same prompts in a batch of 8 or 32 instead of 1 changes the chosen experts for between 2.6%
(layer 0) and 10.6% of words, depending on the layer; 7.25% overall.*

## How routediff works
1. **Learn the normal rounding noise.** Compare the reference with harmless variants (different summation order, with
   or without fused multiply-add) and record, per operation, the largest difference seen; twice that is the tolerance.
2. **Find the first operation outside it.** Dump every intermediate of both backends and compare them in order; the
   first one over its tolerance is the suspect.
3. **Prove it by patching.** Replace that operation's output in the candidate with the reference's and rerun. If the
   expert choices (or the final output) come back, the suspect is confirmed; if not, add the next suspect and repeat.

```mermaid
flowchart LR
  R[reference dumps] --> C[per-op compare vs tolerances]
  K[candidate dumps] --> C
  C --> F[first flagged op]
  F --> P[patch-forward: rerun candidate with that op from the reference]
  P --> V{routes or output restored?}
  V -- yes --> L[localised]
  V -- no --> F
```

## Results

| Question | Answer | Where |
|---|---|---|
| SGLang prefill, batch size alone (preregistered H1 PASS) | OLMoE bf16 on one L4, 200 prompts x 16 layers: batch 8 vs batch 1 changes the expert set of 7.25% of token-layers (95% CI 7.03-7.48%); batch 32: 7.21%. Same config twice: 0. Every sequence has at least one flip. SGLang 0.5.21 + backport of #40980; prefill only. | `findings/census-olmoe-prefill` |
| Does routediff find planted porting bugs? (blinded, preregistered) | Detection 44/44, correct localisation 44/44, false alarms 0/32 benign pairs. Clopper-Pearson: detection >= 0.92, false-alarm rate <= 0.109. The planted bugs are large (250x-10^6x the tolerance), so this validates the mechanics, not the sensitivity floor. | `findings/routediff-eval-real-v2` |
| How subtle a bug can it catch? *(exploratory)* | Narrowing one accumulator at a time on a known, single-site fault: routediff flags a narrowed router or o_proj accumulator down to 21 of fp32's 23 mantissa bits, while route flips start only at <= 18 bits, and localises it correctly in 56/56 cases from 7 to 20 bits. At the detection floor it sometimes points to a downstream op (6 of 11 floor cases). Routes need >= 18 bits in the router accumulator and >= 19 in o_proj; RMSNorm, expert and combine accumulators did not flip this block's routes at any width (expert and combine run after selection, so they cannot). | `findings/accumulator-width` ([chart](findings/accumulator-width/flips_vs_bits.png)) |
| A real discrepancy | The port's original RMSNorm summation order (found by hand in M2): calibrated mode treats it as benign (17 vs 366 scaled ULP, 0 flips); strict mode localises it automatically to `router_input`. | `findings/case-study-rmsnorm` |
| Does the C++ port match PyTorch? | fp32, OLMoE layers 0 and 7: same expert sets on 1348/1348 tokens per layer; bit-exact through the router and the expert gate/up matmuls. The rest differs only after `exp` (2-5 ULP), and port and PyTorch are equally close to fp64 (relative L2 1.06e-6 vs 1.06e-6 at the expert down projection, layer 0; 1.18e-6 vs 1.19e-6 at the output). The preregistered 8-ULP element criterion fails at the expert down projection (~80 ULP p99, from near-cancelling sums); reported as a FAIL. | `findings/m2-postfix-validation` (incl. `oracle_comparison.json`), `findings/m2-initial-layer0` |
| Porting a second model *(exploratory)* | Qwen3-30B-A3B layer 0 in ~15 minutes from export to match (rectangular `o_proj`, renormalised top-k). fp32: same expert sets on 1361/1361 tokens; bit-exact through the router logits. bf16: 9.1% of tokens have an exact tie at the top-k cutoff (vs 1-2% for OLMoE). | `findings/qwen3-port` |
| routediff on a backend I didn't write (demo, not blinded) | Apple GPU (MPS), PyTorch eager bf16, OLMoE layer 0: a forward hook negates `router_logits` and changes 64/64 routes; routediff localises `router_logits`, patching restores the routes, and the repaired run is bit-exact at 13/13 ops. | `findings/torch-patch-forward-bf16/hook-demo-mps-router` |
| PyTorch CPU vs NVIDIA GPU (CUDA, L4) | 64 positions per layer, layers 0 and 7. fp32: 0/64 route differences, nothing flagged. bf16: `router_probs` is the first op over tolerance; after patching it, 1/64 route sets differ (within the bf16 allowance, not exact agreement). Confound: torch 2.13.0+cu130 on the GPU vs 2.14.1 on the CPU, so differences are not attributable to hardware alone. | `findings/backend-pairs-cuda` |
| PyTorch CPU vs Apple GPU (MPS), eager vs grouped_mm | fp32: no route differences, nothing flagged. bf16: every route difference is an exact tie at 8th/9th place. PyTorch CPU on macOS breaks ties by its selection algorithm (libc++ `nth_element`, reproduced exactly), not by a rule; the Linux build of the same torch picks the same set on only 19/38 ties (CI run 38991546221). grouped_mm's combine (fp32 sum, one rounding) is closer to fp64 than eager's on 96.9% / 99.5% of tokens. | `findings/backend-pairs-v1`, `findings/bf16-tolerances` |
| Cross-chip all-reduce of the router input (emulation; preregistered H4 PASS) | fp32 wire: 0 flips at every chip count. bf16 wire, ring: 10 to 49 of 1348 tokens flip (layer 0) from 2 to 64 emulated chips; tree flips about half as many. Real NCCL matches the emulation bit-exactly at 2 ranks; at 4 ranks NCCL's order differs (next row). | `findings/h4-collective-sweep-fp32` ([chart](findings/h4-collective-sweep-fp32/bootstrap/accuracy_vs_bytes.png)), `findings/nccl-validation` |
| Real NCCL's summation order (4x L4 on Modal, NCCL 2.29.7, shared-memory transport; post-hoc, descriptive) | Recovered from the outputs alone, testing all 48 sequential and pairwise rank orders per element: Tree sums ranks sequentially in the order 2, 3, 1, 0 (100% of elements); Ring splits the buffer into 8 equal contiguous blocks, and block b starts at rank (b+1) mod 4 and sums cyclically (1230, 2301, 3012, 0123, repeated), 100% of 1.2M elements in fp32 and bf16. No single fixed order explains more than ~69% of Ring elements, so an emulator must model the per-block rotation. No PCIe/NVLink claim. | `findings/nccl-validation/order-followup` |
| Same, on a bf16 model *(exploratory)* | At 2-8 chips a bf16 wire re-decides 5-19 tokens (0.4-1.4%) vs the single-chip bf16 model; an fp32 wire changes none. | `findings/h4-collective-sweep-bf16-exploratory` |
| SIMD | NEON is 2.6-2.9x faster than plain C++ computing the same summation order (21-23x vs a sequential FMA chain). Its order changes no routes and is ~4x closer to fp64. The block is 15x faster on one thread; the expert matmuls take ~90% of the time. | `findings/simd` |

**Found along the way:** SGLang 0.5.21 OLMoE route-capture crash (fixed upstream in #40980, backported here) and a
capture-buffer sizing overflow in `routed_experts.py` (still on main; documented, not yet reported). On an AMD EPYC
(AVX2) CI runner, PyTorch's SiLU gave different bits for a strided vs a contiguous input of equal values (3 elements),
so a bit-exact claim must state memory layout (`findings/ci-portability`).

> **Limitations**
> - **One model in depth** (OLMoE-1B-7B, MoE blocks of layers 0 and 7), plus one Qwen3 layer. The C++ port covers the
>   MoE block only (no attention, KV cache or RoPE), so routediff is validated only on bugs in those ops.
> - **Emulation:** collectives beyond 2 ranks are emulation; at 4 ranks real NCCL's Ring order rotates per chunk,
>   which the emulator does not model (NCCL's order was recovered post hoc, not built into the emulator). The NCCL check ran
>   on 4x L4 on Modal over shared memory; the interconnect topology was not verified.
> - **Planted bugs are large:** the blinded evaluation does not establish a sensitivity floor; the exploratory width
>   sweep is the only evidence about subtle faults.
> - **The SGLang census** is prefill only, one GPU type, one model, no attribution to ops (no router scores captured).
> - **Small counts:** route flips in the C++ experiments are tens of tokens out of 1348; most comparisons have no
>   confidence intervals (H4 and the census do).
> - **The 8-ULP fidelity criterion failed as preregistered;** the port matches PyTorch in routes and in error vs fp64,
>   not element-wise.

## Quick start
Requires Python 3.12, CMake and a C++17 compiler. Install the full test dependencies
below; the reference tests require the pinned Torch/Transformers runtime, and
Modal's SDK is needed to test job definitions locally. These commands launch no
cloud jobs.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Linux: the CPU wheel used in CI. macOS: the native wheel from PyPI.
if [ "$(uname -s)" = "Linux" ]; then
    python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
else
    python -m pip install torch==2.14.1
fi
python -m pip install -e ".[dev,reference,modal]"
make test   # Python tests + C++ tests; builds port/build/moe_run
make demo   # localises a synthetic router fault; prints the report path
# compare two dump directories, localise the first out-of-tolerance op, confirm by patching:
python -m routediff compare REF_DUMP CAND_DUMP --out report/ --tolerances findings/routediff-eval-real-v2/tolerances.json \
    --split evaluation --rerun 'port/build/moe_run --export EXPORT --numerics N.json --dump {out} {patch}'
```

## Related work
- arXiv 2607.28097: reduction order, accumulator precision and operand conversion in sparse-MoE expert aggregation
  change routes and generated text (DeepSeek-V4-Flash), with a proposed numerical compatibility contract for MoE
  runtimes and hardware backends.
- arXiv 2608.11212: route flips under simulated 4-bit KV-cache quantization on OLMoE-1B-7B, with a causal
  interchange intervention and a router-margin flip detector (AUC 0.772).
- Routing replay: R3 (arXiv 2510.11370) records the inference engine's expert choices and replays them in training,
  after measuring that about 10% of router decisions differ between the training and inference engines on
  Qwen3-30B-A3B. Miles (arXiv 2609.08368) implements R3 as an option in an SGLang-based RL system.
- Fireworks AI blog (2026-09-30): near-ties flipping expert selection, and combine precision mattering after router
  replay, in production RL (no flip rates reported).
- Batch-invariant kernels: Thinking Machines; SGLang `--enable-deterministic-inference`.
- First-divergence tensor comparison across backends: msprobe, Polygraphy, DeepView.
