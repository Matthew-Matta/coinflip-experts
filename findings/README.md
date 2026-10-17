# Findings

One folder per result. Each README starts with a short summary (what I found, why it matters, the main caveat, how to
reproduce), followed by the details. *Preregistered* results follow [PREREG.md](../PREREG.md); *exploratory* ones were
run afterwards; *descriptive* ones measure real systems with no planted ground truth.

| Folder | Result | Type |
| --- | --- | --- |
| **Headline results** | | |
| [census-olmoe-prefill](census-olmoe-prefill/) | In SGLang, batch 8 vs batch 1 changes the experts for **7.25%** of token-layers; the same run repeated: 0% | preregistered |
| [routediff-eval-real-v2](routediff-eval-real-v2/) | routediff found and pinpointed **44/44** planted porting bugs, **0/32** false alarms (blinded) | preregistered |
| [accumulator-width](accumulator-width/) | Router accumulator needs **≥ 18** mantissa bits, router input **≥ 19**; routediff flags 3-4 bits earlier | exploratory |
| [nccl-validation/order-followup](nccl-validation/order-followup/) | NCCL's 4-GPU summation order recovered from its outputs, **100%** bit-exact (tree 2→3→1→0, ring rotates per block) | descriptive |
| [qwen3-port](qwen3-port/) | Second model (Qwen3-30B-A3B layer) ported in ~15 min: **1361/1361** routes match PyTorch | exploratory |
| **C++ port vs PyTorch** | | |
| [m2-initial-layer0](m2-initial-layer0/) | First port failed the preregistered fidelity check; cause found (RMSNorm summation order) | preregistered |
| [m2-postfix-validation](m2-postfix-validation/) | After the fix: 1348/1348 routes match, bit-exact through the router; element-wise criterion still fails after `exp` | descriptive |
| [case-study-rmsnorm](case-study-rmsnorm/) | routediff's strict mode finds that RMSNorm bug automatically; calibrated mode rates it benign | descriptive |
| [simd](simd/) | NEON SIMD is 2.6-2.9x faster than the same summation order in plain C++; 0 route changes | descriptive |
| [qwen3-layer0](qwen3-layer0/) | Qwen3 layer export (one 4 GB shard of 16, 1361 tokens), PyTorch mirror bit-exact | exploratory |
| **Other backends** | | |
| [backend-pairs-v1](backend-pairs-v1/) | CPU vs Apple GPU, eager vs grouped_mm: fp32 identical routes; bf16 differences are all exact ties | descriptive |
| [backend-pairs-cuda](backend-pairs-cuda/) | CPU vs NVIDIA L4: fp32 0/64 route differences; bf16 1/64 after patching | descriptive |
| [topk-portability](topk-portability/) | On exact ties, PyTorch picks the same experts on Linux and macOS only 19/38 times | descriptive |
| [bf16-tolerances](bf16-tolerances/) | bf16 tolerances; exact ties inflate the output tolerance 48x | exploratory |
| [torch-patch-forward-bf16](torch-patch-forward-bf16/) | routediff's patch-forward on PyTorch backends via forward hooks; null patch bit-exact | exploratory |
| […/hook-demo-mps-router](torch-patch-forward-bf16/hook-demo-mps-router/) | Demo: router fault on the Apple GPU changes 64/64 routes; localised and repaired bit-exactly | exploratory |
| […/hook-demo-cpu](torch-patch-forward-bf16/hook-demo-cpu/) | Demo: combine fault on CPU localised; repaired run bit-exact on 13/13 ops | exploratory |
| [ci-portability](ci-portability/) | On an AMD EPYC CI runner, SiLU gives different bits for strided vs contiguous input (3 elements) | descriptive |
| **Multi-chip** | | |
| [h4-collective-sweep-fp32](h4-collective-sweep-fp32/) | bf16 on the wire flips 10 to 49 of 1348 routes from 2 to 64 emulated chips; fp32 flips none | preregistered |
| [h4-collective-sweep-bf16-exploratory](h4-collective-sweep-bf16-exploratory/) | Same sweep on a bf16 model: 5-19 tokens re-decided at 2-8 chips | exploratory |
| [nccl-validation](nccl-validation/) | Real NCCL on 4x L4 matches the emulation bit-exactly at 2 GPUs, not at 4 | descriptive |
| **Tooling** | | |
| [offline-demo](offline-demo/) | `make demo`: localises and repairs a synthetic router fault, no GPU or model download | descriptive |
| [charts](charts/) | Index of the charts used in the README | descriptive |
