# CPU / CUDA backend pairs (descriptive)

OLMoE layers 0 and 7, eager pinned, **64 identical captured positions per layer**:
first two positions from each original prompt (32 prompts; calibration/evaluation
positions both included). CPU references were recomputed at this same shape,
not sliced from a larger-batch computation. No new tolerances were calibrated.

CUDA: **NVIDIA L4**, driver **610.57.04**, torch **2.13.0+cu130**, CUDA **13.0**,
Transformers **5.12.1**. CPU: Apple M1 Pro, torch **2.14.1**, Accelerate,
Transformers 5.12.1. These pairs differ in hardware and software build; do not
attribute a discrepancy to hardware alone. TF32 disabled for CUDA dump generation.

| Layer | Dtype | Initial route-set differences / 64 | Earliest flagged op | Patch result |
| --- | --- | --- | --- | --- |
| 0 | fp32 | 0 | none | No patch requested; no op exceeds real-v2 tolerances |
| 7 | fp32 | 0 | none | No patch requested; no op exceeds real-v2 tolerances |
| 0 | bf16 | 0 | router_probs | Single-site patch clears all flags; output criterion passes; **1/64** route sets differ afterwards |
| 7 | bf16 | 1 | router_probs | Single-site patch clears all flags; output criterion passes; **1/64** route sets differ afterwards |

All **four unpatched CUDA mirrors equal the actual Transformers module
bit-exactly**. A same-CUDA router-logit null patch leaves **all 13 operations
bit-exact** in every pair. Full patch-forward uses the torch forward-hook adapter.
The reports retain every per-op metric and patch check.

## Interpretation limits

`router_probs` is the earliest op exceeding the frozen tolerance, **not the first
bitwise difference or proof of a faulty softmax kernel**. The bf16 router logits
already differ below their tolerance. Both combined outputs were also within the
broad bf16-exploratory output tolerance **before** patching (worst scaled ULP 7.0
and 71.5, threshold 482). This is a descriptive threshold-localisation result,
not a demonstration that an initially unacceptable combined output was repaired.

The frozen bf16 route allowance is 0.017543859649122806, or about 1.123 tokens
on 64 positions. Consequently the tool's `routes_restored=true` means within
that allowance, **not exact route agreement**. Patching introduces one route-set
difference on layer 0; the one difference on layer 7 remains. We do not claim
bit-exact CPU/CUDA restoration or score these real pairs as planted bugs.
The separate known-fault MPS demo supplies the exact route-affecting repair demo.

## Provenance and reproduction

`settings.json` records the selected positions, frozen input hashes and runtime
source hashes. `summary.json` records library versions, memory and runtime.
GPU memory peaks are PyTorch allocated bytes in the parent dump/null-patch process;
patch subprocess peaks were not captured. Full raw dumps, null patches and job
metadata are at `data/followup/downloaded/84cdd47f1494/cuda-pairs/` locally.

The shared NCCL/CUDA job completed PASS with no GPU retry. It ran 150.228 seconds
before archiving, estimating $0.15335 at the configured resource rates; startup,
archiving and transfer are additional, and this is not an audited billing total.
The allocation had a 900-second timeout, two-core / 48-GiB caps and a $0.918684
timeout resource estimate, leaving reserve within the approved $2 total budget.
The app stopped on completion. Earlier upload-only apps never called the GPU function.

```sh
python -m reference.prepare_followup --out data/followup/input-v2
modal run modal/numerics_followup.py                         # dry run
# New explicit spending approval is required to reproduce the paid allocation:
modal run modal/numerics_followup.py --gpu-approved
```

The paid job restores weights from the existing pinned checkpoint cache and
requires their original SHA-256 values before any measurement. No fresh model
download or numerical tuning is performed. Output archives carry a SHA-256.
