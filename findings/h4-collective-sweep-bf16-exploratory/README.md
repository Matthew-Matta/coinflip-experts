# Collective sweep on the bf16 base (EXPLORATORY, after preregistration)

**Label: exploratory**, per PREREG amendment c18d512, which made `hf_fp32` the primary H4 base. The primary result is
`findings/h4-collective-sweep-fp32` (PASS on both layers). **This is reduction-order emulation** on one CPU, not a
measurement of an interconnect. The exact order emulated is in `REDUCTION_ORDER.md`. Real NCCL matches it bit-exactly
at 2 ranks but not at 4 (`findings/nccl-validation`), so for n >= 4 this describes a sum order of the stated kind,
not NCCL's.

**Setup.**
- Base numerics `hf_bf16`: bf16 storage and activations, fp32 accumulators, bf16 router logits, fp32 softmax, combine
  in a bf16 buffer. These match transformers' eager bf16.
- OLMoE-1B-7B layers 0 and 7, 1348 tokens each, compared with the fp64 oracle.
- Binary: `port/build/moe_run` sha256 `c30b6f2f…`, the same binary as the fp32 sweep and real-v2.
- Run: `python -m port.sweeps.collective_sweep --base hf_bf16 --threads 4 --out data/results/h4-collective-sweep-bf16-exploratory`,
  started 2026-10-17 ~01:10, wall time 11.1 h (laptop shared with other jobs and asleep part of the night).
- This is a new run. The earlier partial bf16 run (started 2026-10-16 18:14, killed, never examined) stays unexamined
  in `data/results/h4-sweep-bf16-partial-unexamined`.

**Metric.** The bf16 base already routes some tokens differently from fp64 without any collective (n = 1): 11 / 1348
on layer 0 and 19 / 1348 on layer 7. Two numbers are therefore reported:
- *excess vs oracle* = flips vs oracle at n, minus flips vs oracle at n = 1 (the net harm of the collective);
- *changed vs n = 1* = tokens whose expert set differs from the no-collective bf16 run (how many near-ties the
  collective re-decides, in either direction).

## Router-input (o_proj) all-reduce: route flips, of 1348 tokens

| layer 0 | bf16 wire, ring: excess / changed | bf16 wire, tree: excess / changed | fp32 wire (ring and tree) |
|---|---|---|---|
| n = 2 | +3 / 9 | +3 / 9 | 0 / 0 |
| n = 4 | +11 / 19 | +7 / 19 | 0 / 0 |
| n = 8 | +6 / 10 | +5 / 13 | 0 / 0 |
| n = 16 | +12 / 19 | +10 / 18 | 0 / 0 |
| n = 32 | +22 / 32 | +10 / 24 | 0 / 0 |
| n = 64 | +37 / 45 | +16 / 26 | 0 / 0 |

| layer 7 | bf16 wire, ring: excess / changed | bf16 wire, tree: excess / changed | fp32 wire (ring and tree) |
|---|---|---|---|
| n = 2 | 0 / 5 | 0 / 5 | 0 / 0 |
| n = 4 | -5 / 11 | -1 / 9 | 0 / 0 |
| n = 8 | +3 / 10 | -2 / 8 | 0 / 0 |
| n = 16 | +1 / 12 | -4 / 18 | 0 / 0 |
| n = 32 | +1 / 13 | +2 / 10 | 0 / 0 |
| n = 64 | -1 / 19 | -1 / 11 | 0 / 0 |

- **n = 2-8, the range a real tensor-parallel group spans:** a bf16 wire re-decides 5-19 tokens (0.4-1.4%) relative
  to the same bf16 model without a collective. Net harm vs fp64 is small: +3 to +11 tokens on layer 0, -5 to +3 on
  layer 7.
- **An fp32 wire changes nothing:** 0 tokens at every n, on both layers and both topologies. The fp32 all-reduce is
  invisible in a bf16 model, so the cost of a bf16 wire is entirely the wire.
- **Layer 0:** harm grows with n for ring (rho 0.94 over n = 2..64) and less for tree (+16 at 64 vs +37 for ring), as in
  the fp32-base sweep.
- **Layer 7:** the bf16 wire re-decides as many near-ties toward the fp64 answer as away from it, so net harm stays
  near zero (Spearman rho 0.12), while the number of changed tokens still grows (ring 5 to 19). Whether a collective
  "hurts" depends on whether you count against fp64 or against the single-chip port. For porting, the second matters:
  a multi-chip port of a bf16 model is not route-identical to its single-chip version unless the wire is fp32.

## Down-projection all-reduce (cannot change this block's routes)
Output relative L2 vs oracle:
- Layer 0: no collective 1.51e-2. bf16 wire 1.51e-2 at n = 2 rising to 1.68e-2 (ring) / 1.53e-2 (tree) at n = 64.
  fp32 wire 1.51e-2 at every n.
- Layer 7: 2.18e-2 rising to 2.34e-2 / 2.21e-2; fp32 wire unchanged.
- The bf16 storage error dominates; the wire adds at most ~11% (ring, 64 emulated chips).

## About `h4_verdict.json`
The script applies the preregistered H4 test mechanically; it was written for the fp32 base. With a bf16 base,
criterion (b), "fp32 wire flat", is trivially met at the base's own flip level, and (a) gives layer 0 PASS, layer 7
FAIL. **This is not a preregistered test** and is not reported as an H4 result.

## Limitations
- Counts are small and there are no confidence intervals (the fp32-base sweep has prompt-cluster CIs, from Codex).
- Two layers, one model.
- Ring rotation and tree shape are conventions (`REDUCTION_ORDER.md`).
