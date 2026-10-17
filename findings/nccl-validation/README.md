# Real sums versus my emulator

**In one line:** My emulator matched **8/8** configurations with two GPUs and **0/8** with four.

- **What I found:** I compared real NCCL sums with my unchanged ring and tree emulation on captured model inputs.
- **Why it matters:** A shared algorithm name does not guarantee the same rounded sum.
- **Caveat:** I could not verify the interconnect topology; this checks sums, not route changes.

**Type:** descriptive   ·   **Reproduce:** `modal run modal/gate_olmoe.py --mode nccl --gpu-approved`

---

## Details

**Two ranks: 8/8 configurations bit-exact. Four ranks: 0/8 bit-exact.** Numerical mismatches are valid results; I did not tune algorithms or inputs after observing them. Execution completed successfully, which is not a claim that the numerical emulation matches NCCL.

Four NVIDIA L4 GPUs; torch 2.13.0+cu130; CUDA 13.0; NCCL **2.29.7**. Each world size (2 or 4) uses a new torchrun process group with `NCCL_ALGO=Ring` or `Tree` and default protocol selection. GPU-side collective phase wall time: 41.1 seconds; allocation/startup overhead is additional and not an audited billing total.

## Inputs and reference

64 real captured positions from OLMoE layer 0: first two valid positions per each of the original 32 prompts. No synthetic random inputs. Partial o_proj and expert-down dots are computed from the committed scalar-FMA port and its clean fp32 expert activations. Each partial is rounded once to the chosen wire format before NCCL sums it. I independently checked all sixteen expected reductions bit-for-bit against actual `moe_run` collective configs (`moe_validation.json`). Build: isolated HEAD `9dd291de4a84f21f6c468f0f66f218ce72298f8f`, binary SHA256 `c30b6f2faee45cb23ada7d8237bd5ca544a5bb59c8d1ed2d7fb3dac420c55a1b`.

NCCL reduces the full flattened captured tensor; the port emulates a row-wise ring/tree order. NCCL_ALGO names do not specify rank order, channel/chunk assignment, protocol or intermediate rounding. A match is not guaranteed by the shared name. This is a validation of reduction sums on a captured sample, not of route flips or a full tensor-parallel model.

## Rank-0 results

| n | Algorithm | Target | Wire | Bit-exact | Different / elements | Max reference ULP |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Ring | oproj | fp32 | yes | 0/131072 | 0 |
| 2 | Ring | oproj | bf16 | yes | 0/131072 | 0 |
| 2 | Ring | down | fp32 | yes | 0/1048576 | 0 |
| 2 | Ring | down | bf16 | yes | 0/1048576 | 0 |
| 2 | Tree | oproj | fp32 | yes | 0/131072 | 0 |
| 2 | Tree | oproj | bf16 | yes | 0/131072 | 0 |
| 2 | Tree | down | fp32 | yes | 0/1048576 | 0 |
| 2 | Tree | down | bf16 | yes | 0/1048576 | 0 |
| 4 | Ring | oproj | fp32 | no | 41399/131072 | 12288 |
| 4 | Ring | oproj | bf16 | no | 43738/131072 | 3.32307e+35 |
| 4 | Ring | down | fp32 | no | 328970/1048576 | 1.04858e+06 |
| 4 | Ring | down | bf16 | no | 327421/1048576 | 2.65846e+36 |
| 4 | Tree | oproj | fp32 | no | 40060/131072 | 6144 |
| 4 | Tree | oproj | bf16 | no | 41666/131072 | 3.32307e+35 |
| 4 | Tree | down | fp32 | no | 306549/1048576 | 1.04858e+06 |
| 4 | Tree | down | bf16 | no | 306765/1048576 | 1.32923e+36 |

All ranks report the same element counts and maximum ULP for each configuration. ULP is measured in the reference wire dtype. Cancellation can leave exact zero in the emulation: its ULP is the subnormal spacing (2^-133 in bf16), so a small nonzero NCCL result can produce an enormous maximum ULP. These maxima should not be read as relative error or route impact.

## Interconnect limitation

`nvidia-smi topo -m` was attempted and returned exit **255**, “Failed to run topology matrix”. The PCI bus query returned “[Unknown Error]” for each GPU. Both outputs are preserved in `summary.json`; **PCIe/NUMA/NVLink topology could not be verified inside this Modal container**. I cannot claim a measured PCIe topology. NCCL debug logs confirm the requested `NCCL_ALGO` and show channel transport `SHM/direct/direct` (shared-memory staging); direct GPU P2P was unavailable. This is not a direct PCIe-P2P validation. An initial invocation stopped at this metadata failure before reductions; the completed invocation retained the failure and performed the unchanged numerical experiment.

## Reproduce

```sh
python -m reference.prepare_nccl --out data/nccl  # fresh output only; committed build required
modal run modal/gate_olmoe.py --mode nccl   # no-GPU dry run
# explicit GPU budget approval required
modal run modal/gate_olmoe.py --mode nccl --gpu-approved
```

Prepare the `data/nccl/partials` arrays with `reference/collective_partials.cpp`, linked against the isolated committed port headers/library, and retain the selection/checksums. The modal runner uploads only these frozen partials and expected sums, never full model weights. Raw NCCL logs and rank reports are retained locally in `data/nccl/remote`; input SHA256 hashes are in the summary.
