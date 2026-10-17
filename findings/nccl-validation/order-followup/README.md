# Saved-rank NCCL follow-up: order and position (descriptive)

**4xL4 on Modal, SHM transport.** All four NCCL logs confirm
`SHM/direct/direct`; no PCIe or NVLink topology claim. Torch 2.13.0+cu130,
CUDA 13.0, NCCL 2.29.7, NVIDIA driver 610.57.04 (`run.json`).

The rerun used the original 24 frozen partial/expectation arrays, unchanged
world sizes, algorithm settings, dtypes and collective call order. It saved
all per-rank outputs as float32 `.npy` files (bf16 values remain exactly
representable). Every archived output hash was verified locally. All ranks
are bit-identical within each configuration. Raw files are under
`data/followup/downloaded/84cdd47f1494/nccl/`, also linked locally at
`data/nccl/remote/followup-84cdd47f1494/`.

## Results

Two ranks reproduce the port in **8/8 configurations**; four ranks reproduce it
in **0/8**. Mismatch counts and maximum reference ULP repeat the original
aggregate results. The original tensors were not saved, so the order analysis
below describes the new captured outputs, not recovered historical tensors.
The port and frozen expectations were not tuned.

For **each element**, all 24 permutations of the four partials were tested with
sequential and balanced-pairwise summation (48 labelled candidates). Each add
rounds in the stated wire dtype. BF16 arithmetic is checked against explicit
Torch additions for all 48 candidates. Equivalent permutations can overlap;
match shares do not uniquely identify executed instructions.

| Algorithm | Target | Wire | Best fixed order | Share explained | Any tested order | Max size-scaled error (wire eps units) |
| --- | --- | --- | --- | --- | --- | --- |
| Ring | o_proj | fp32 | sequential 2301 | 68.6218% | 100% | 1.854010 |
| Ring | o_proj | bf16 | sequential 1230 | 67.3393% | 100% | 1.607516 |
| Ring | down | fp32 | sequential 1230 | 68.7964% | 100% | 1.886671 |
| Ring | down | bf16 | sequential 1230 | 68.8304% | 100% | 1.946681 |
| Tree | o_proj | fp32 | sequential 2310 | 100% | 100% | 1.499068 |
| Tree | o_proj | bf16 | sequential 2310 | 100% | 100% | 1.340753 |
| Tree | down | fp32 | sequential 2310 | 100% | 100% | 1.497648 |
| Tree | down | bf16 | sequential 2310 | 100% | 100% | 1.610696 |

Rank IDs are zero-based. Sequential 2310 means `((p2+p3)+p1)+p0`;
3210 is equivalent and also matches all Tree elements. A graph called Tree
need not reduce four rank contributions as the port's balanced 0123 tree.
Every candidate's share is retained in `order_shares.csv`; per-element matching
multiplicities and rank equality checks are in `orders.json`.

## Element position

No fixed candidate explains every Ring element. Dividing the C-order flattened
output into **eight equal contiguous blocks**, the following sequential orders
reproduce **all elements**, for both targets and both wire types:

```text
block:  0     1     2     3     4     5     6     7
order:  1230  2301  3012  0123  1230  2301  3012  0123
```

Block widths are 16,384 elements for o_proj and 131,072 for down. This is a
**post-hoc descriptive positional fit**, not an emulator modification or a
verified mapping to NCCL chunks/channels. The 64-bin comparison, full candidate
shares by position and quarter, and exact block-fit checks are retained in
`position_shares.csv`, `orders.json` and `position-pattern.json`. The best fixed
Ring order has bin shares ranging roughly 51–56% to 100%; Tree's matching fixed
order stays at 100% in every bin. Thus an order consistent with the data varies
with element position for Ring, while the same fixed Tree order suffices here.

## Size-aware error

`errors.csv` adds the requested magnitude-normalised error:

```text
abs(NCCL - emulation) / sum(abs(partials)) / eps_wire
```

Here `eps_wire` is 2^-23 for fp32 and 2^-7 for bf16: ULP spacing at 1, rather
than the ULP of a possibly cancelled output or of the magnitude sum. The table
reports the maximum over elements; p99 and unscaled relative errors are also
retained. All n=2 values are zero. This avoids interpreting the enormous
reference-ULP maxima near cancellation (up to ~2.66e36) as enormous relative
errors. It does not measure route impact.

## Execution and reproduction

One combined NCCL/CUDA allocation completed PASS and stopped: 150.228 seconds
before archiving, estimated resources $0.15335; startup, archiving and transfer
are additional, not an audited bill. Timeout resource estimate was $0.918684
(900 seconds, zero retries, CPU/RAM limits), leaving reserve under the approved
$2 total. Upload-only apps stopped before any GPU function call; only one GPU
function executed. The archive SHA is recorded in `run.json`.

```sh
python -m analysis.nccl_followup --source data/followup/downloaded/84cdd47f1494 --partials data/nccl/partials --out data/followup/order-analysis
```

The runner verifies input/output hashes and analyses every distinct rank output.
Local parquet and full provenance remain under `data/followup/order-analysis/`;
public summaries retain source hashes without copying local host metadata.
