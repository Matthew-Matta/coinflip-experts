# SIMD (NEON) path: speed and numerics (PREREG 3c, descriptive)

Single thread, Apple M1 Pro (8 performance + 2 efficiency cores). Speed numbers are a median of 20 timed runs on the
same data. "Scalar" is the port's default fp32 dot product: one sequential FMA chain, the order it emulates.

## What the SIMD path is
An fp32 dot product with `accs` NEON accumulators (4 or 8) of 4 lanes each, using FMA. The accumulators are summed as a
balanced tree, ((a0+a1)+(a2+a3)) for accs 4, and the lanes by FADDP, ((l0+l1)+(l2+l3)). The scalar tail is added last.
A portable loop in exactly the same order backs it. Tests show NEON == portable bit-exactly (including tails) for
accs 1, 2, 4 and 8, and 1-accumulator SIMD == `Order::Strided(4)` (`port/tests/test_moe.cpp`).

## Speed (fp32 matvec, 2048 columns)

Canonical run: run 4, binary `bench_simd` sha256 `ebd4374d…`.

| regime | weights | accs | sequential FMA chain | same order, plain C++ | NEON | **NEON vs same-order C++** | NEON vs sequential chain | NEON GFLOP/s | weight stream |
|---|---|---|---|---|---|---|---|---|---|
| L2-resident | 8 MiB (1024 rows) | 4 | 4.053 ms | 0.489 ms | 0.179 ms | **2.73x** | 22.7x | 23.5 | 46.9 GB/s |
| DRAM-bound | 256 MiB (32768 rows) | 4 | 126.4 ms | 16.95 ms | 5.949 ms | **2.85x** | 21.2x | 22.6 | 45.1 GB/s |
| L2-resident | 8 MiB | 8 | 3.923 ms | 0.336 ms | 0.181 ms | **1.86x** | 21.7x | 23.2 | 46.4 GB/s |
| DRAM-bound | 256 MiB | 8 | 134.6 ms | 10.80 ms | 5.910 ms | **1.83x** | 22.8x | 22.7 | 45.4 GB/s |

- **Headline:** explicit NEON is **2.6-2.9x faster than plain C++ that computes the same summation order** (accs 4),
  so the numerics are identical and only the code generation differs. The 21-23x figure is relative to a *sequential
  FMA chain*, a deliberately latency-bound baseline (~1 GFLOP/s) that emulates a sequential reduction order. It is
  not an optimised scalar kernel. Most of that factor (7.5-8.3x) comes from breaking the dependency chain, not from
  SIMD.
- **What limits it (corrected).** With 4 accumulators and 4-cycle FMA latency, FMA latency caps the kernel at one
  vector FMA per cycle, about 25.8 GFLOP/s at 3.2 GHz. We measure 22.6-23.5, so the earlier word "compute-bound" was
  wrong. The 8-accumulator test then shows FMA latency is *not* the binding limit either: doubling the independent
  chains lifts the latency cap to ~51 GFLOP/s, yet throughput stays at 22.7-23.2 GFLOP/s.
  - The second ceiling at about one FMA per cycle is load issue. Each vector FMA needs two 16-byte loads (weights and
    activations), and we measure ~46 GB/s of weights plus the same again of activations, about two vector loads per
    cycle. This is an inference, not a measurement of the load ports.
  - Memory size does not matter: the L2-resident and DRAM-bound runs stream at the same rate, so DRAM is not the
    limit for one core.
  - **No bandwidth claim.** A kernel that reuses activations across rows (register blocking) would be the next step.
    It was not attempted (time box).
- With 8 accumulators, the plain-C++ same-order loop is itself 1.4-1.6x faster than with 4. That is why NEON's
  advantage over it shrinks to ~1.85x while NEON's absolute speed is unchanged.
- **Machine state.** The gate before each run was 3 checks, 20 s apart, with no project jobs (moe_run, evaluation,
  routediff, sweeps, pytest, compilers) and total CPU of everything else < 250% (under 2.5 of 10 cores). The rest was
  desktop apps (browser, terminal, window server).
  - Recorded per run in `bench_*-accs*.machine.txt`: background was 1.3-2.0 cores before each run and 1.2-3.6 cores
    right after.
  - The 1-minute load average before each run was 4.7-12.6 in run 4 and 2.3-4.4 in run 3. In run 4 it partly lags a
    parallel build that had finished about 1 minute earlier; the rest of the queue was not identified. %CPU sampled
    at the moment of each run stayed below 2.1 cores.
  - Runs 3 and 4 give 2.64x / 2.73x (L2-resident) and 2.60x / 2.85x (DRAM-bound) for NEON vs same-order C++ (accs 4).
    The headline is therefore stated as a range, 2.6-2.9x.
  - The benchmark is single-threaded, so it had a performance core to itself, but this is a quiet machine, not an
    idle one.

### Earlier runs (kept, labelled)
- `run1_*` (local only, not committed): first pass, accs 4, load average not settled.
- `run2_possibly-contended_*`: accs 4. These overlapped routediff backend-pair compares (files written 20:17-20:20),
  so they were *not* on an idle machine.
- `run3_quiet_bin-fc600db0_*`: same quiet-machine gate, previous binary.
  - accs 4: 2.64x / 2.60x vs same order, 22.3x / 23.3x vs sequential chain, 23.9 / 23.0 GFLOP/s.
  - accs 8: 0.39x / 0.40x, because the 8-accumulator NEON path then ran through a generic runtime-indexed loop with
    poor code generation. That was fixed with an unrolled 8-accumulator path in the same order (tests unchanged and
    passing), which is run 4 above.
- `discarded_overhead-confounded_bench_L1-resident-*` (local only, not committed): an attempt at an L1-sized matrix (4 rows). Per-call overhead
  of `matvec` (activation copy per call) dominates at that size, so it says nothing about the kernel. Not used.
- For accs 4, runs 1-3 agree with run 4 within ~10%.

## Where the time goes in one block (bottleneck profile)
`moe_run --profile`: wall time per stage, single thread, layer 0, first 128 tokens, fp32 (`profile_layer0_128tok_1thread_*.json`).

| | scalar (sequential FMA) | NEON accs 4 | NEON accs 8 |
|---|---|---|---|
| block time per token | 106.6 ms | 7.0 ms | 6.4 ms |
| expert gate + up matmuls | 61.6% | 57.1% | 55.6% |
| expert down matmul | 30.1% | 30.9% | 32.0% |
| o_proj matmul | 7.7% | 6.7% | 7.1% |
| everything else (RMSNorm, router, softmax, top-k, SiLU, combine) | 0.6% | 5.3% | 5.3% |

- The block is 15x faster with NEON (accs 4) on one thread. The expert matmuls are 87-92% of the time in every
  configuration: 8 experts x (2 x 1024 x 2048 + 2048 x 1024) multiply-adds per token stream 8 x 24 MiB of fp32
  weights.
- After SIMD, the non-matmul ops become ~5%. The router matmul (64 x 2048) is below 0.3%.
- The bf16-storage preset (`hf_bf16`, scalar) costs the same as fp32 scalar (107.2 ms per token). Rounding to bf16 is
  cheap next to the matmuls.

## Numerics (fp32, real layers, 1348 tokens; fp64 oracle)

| layer | route flips vs oracle (scalar / accs 4 / accs 8) | flips between any two | output rel L2 vs oracle (scalar / accs 4 / accs 8) |
|---|---|---|---|
| 0 | 0 / 0 / 0 | 0 | 1.18e-06 / 2.76e-07 / 2.47e-07 |
| 7 | 0 / 0 / 0 | 0 | 1.26e-06 / 3.42e-07 / 2.94e-07 |

- The SIMD orders change no expert choices. They are 4-5x closer to the fp64 oracle than the sequential order (many
  short partial sums lose less than one long chain). They differ from scalar bit-wise on every token, and accs 4 and
  accs 8 differ from each other on every token (rel L2 3.0e-7 / 3.6e-7), as expected for different orders.
- **routediff, calibrated:** scalar reference vs SIMD candidate, evaluation split, frozen real-v2 tolerances sha256
  `1944f35d…`. Nothing is flagged for accs 4 or accs 8 on either layer: both orders are inside the benign envelope.
- **routediff, strict (supplementary):** localises to `o_proj_out`, with sites `o_proj_out, router_logits, expert_gate,
  expert_up, expert_down`. These are exactly the five matmuls the SIMD path replaces, for accs 4 and accs 8 on both
  layers, and patching them restores the output. This is correct attribution against a fully known ground truth.

## Provenance
- Current binaries: `port/build-simd/bench_simd` sha256 `ebd4374d9bb4229a55e5751c13a997e30185590ccdfe5dc3c8c24a54b13daccf`,
  `port/build-simd/moe_run` sha256 `ccd7537151d6a308acb9eea741930981a3147f988c16d405877dd038cfc94f86`. Built from the
  working tree before the SIMD, e8m-format and profile changes were committed (fcfa361).
- The current `moe_run` reproduces the earlier scalar and accs-4 SIMD dumps 13/13 ops bit-identically on both layers.
  With SIMD off, those reproduce the post-fix dumps from `port/build/moe_run` (sha256 `c30b6f2f…`). So differences
  come from the SIMD order alone.
- accs-4 dumps were made with the earlier `moe_run` (sha256 `0c420ea9…`). accs-8 dumps, routediff accs-8 reports and
  the profiles were made with `ccd75371…`.
- Oracle: `data/results/backend-pairs-analysis/oracle-layer{0,7}`.
