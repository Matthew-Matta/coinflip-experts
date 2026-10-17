# The reduction order the collective emulation uses (for comparison against NCCL)

Source: `port/src/numerics.cpp`, `matvec(..., const Collective& c)`; it is called from `port/src/block.cpp` for
`o_proj` (row-parallel; output = router input after residual + RMSNorm) and for each expert's `down` projection.
This is **reduction-order emulation** on one CPU: no interconnect, no NCCL code, no timing. Everything below is per
output element y[r] of an [rows, K] x [K] matvec.

1. **Sharding of K.** K (o_proj: 2048 attention features in OLMoE; down: 1024 intermediate features) is split into
   n contiguous shards of `ceil(K / n)` columns: chip k owns columns `[k*ceil(K/n), min((k+1)*ceil(K/n), K))`. For the
   n used here (2-64), K/n divides exactly, so every shard has K/n columns.
2. **Local partial per chip.** Each chip computes the dot product over its shard with the base accumulator settings
   (both bases: fp32 accumulator, sequential order, fused multiply-add per step, starting from 0). The partial is then
   **rounded once to the wire dtype** (bf16 or fp32, round-to-nearest-even).
3. **Cross-chip sum, all in the wire dtype** (every addition rounded to the wire dtype):
   - **Ring** (reduce-scatter order). The output rows are divided into n contiguous chunks; row r is in chunk
     `j = floor(r * n / rows)`. Its sum starts at chip `(j + 1) mod n` and adds the partials of chips
     `(j+1), (j+2), ..., (j+n) mod n` one at a time: `acc = 0; acc = rnd(acc + p[(j+1)%n]); ...`. The last partial added
     is chip j's own (the chunk's owner). So the n partials are summed sequentially, in a rotation that depends on the
     row's chunk.
   - **Tree.** A recursive-halving (pairwise) sum over chips 0..n-1 in index order: `sum(lo, hi) = rnd(sum(lo, mid) +
     sum(mid, hi))` with `mid = lo + (hi - lo) / 2`; a single chip's partial is passed through `rnd` (a no-op, since it
     is already in the wire dtype).
4. **Result.** The reduced value is rounded to the model's storage dtype (fp32 for hf_fp32, bf16 for hf_bf16).
   The all-gather that would follow a reduce-scatter only moves data, so it is not emulated. Every chip ends with the
   same value.
5. n = 1 means no collective: one fp32 dot product over all of K, rounded to the storage dtype.

## Against real NCCL
Codex's measurement, `findings/nccl-validation`: 4x L4, NCCL 2.29.7, 64 captured positions, the same rounded partials.
With 2 ranks NCCL Ring and Tree match this emulation bit-exactly (8/8 configurations). With 4 ranks they do not
(0/8; e.g. 41,399 of 131,072 o_proj elements differ for fp32 Ring). Partly this is expected: NCCL reduces the
flattened tensor with its own chunk / channel / rank assignment, which this emulation does not model. **So results for
n >= 4 describe the effect of a sum order of the stated kind, not NCCL's exact order.**

## Known differences from a real NCCL all-reduce (not emulated)
- NCCL's chunking, pipelining and protocols (Simple / LL / LL128) are not modelled. They do not change the order in
  which an element's partials are added, unless the implementation reorders them; that is the thing to check.
- NCCL "tree" is a double binary tree spanning nodes, with a chain inside a node, not a single pairwise tree over
  ranks in index order.
- How NCCL accumulates bf16 (whether each step rounds to bf16 or accumulates wider) is **assumed** to be "round to the
  wire dtype after every addition". This is the main thing to verify.
- The ring start chip is a convention, `(j + 1) mod n` for chunk j. A real ring's start may differ by a rotation, which
  changes the order but not the number of roundings.
