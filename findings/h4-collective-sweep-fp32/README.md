# H4 collective sweep (primary, fp32 base): PASS on layers 0 and 7

**Emulation** (PREREG section 3), not a measurement of any interconnect. Base numerics `hf_fp32` (PREREG amendment c18d512), compared with the fp64 oracle. Margin-free: H4 has its own preregistered criteria.

- Binary: `port/build/moe_run` sha256 `c30b6f2faee45cb23ada7d8237bd5ca544a5bb59c8d1ed2d7fb3dac420c55a1b` (unchanged since 18:13:58; matches the verifier's record). Started started 10-16 18:53:35 at commit 5dc67ef, PREREG amendment c18d512 committed 18:35:22; wall time 80 min.
- Provenance git state is captured at save time (`cd74858`, dirty: uncommitted SIMD source edits that did not affect this binary).

## Verdict (PREREG H4)
- layer0: **PASS**. (a) bf16 ring, Spearman rho over n = 2..64: 0.899 (>= 0.5). (b) fp32 wire: ring pass (rule of three, bound 0.0022, max rate 0.0000); tree pass (rule of three, bound 0.0022, max rate 0.0000)
- layer7: **PASS**. (a) bf16 ring, Spearman rho over n = 2..64: 0.880 (>= 0.5). (b) fp32 wire: ring pass (rule of three, bound 0.0022, max rate 0.0000); tree pass (rule of three, bound 0.0022, max rate 0.0000)

## Layer 0: route flips vs fp64 oracle from the router-input (o_proj) all-reduce
Tokens: 1348. No collective (n = 1): 0 flips.

| n | bf16 ring | bf16 tree | fp32 ring | fp32 tree |
|---|---|---|---|---|
| 2 | 10 | 10 | 0 | 0 |
| 4 | 15 | 10 | 0 | 0 |
| 8 | 18 | 16 | 0 | 0 |
| 16 | 15 | 21 | 0 | 0 |
| 32 | 34 | 18 | 0 | 0 |
| 64 | 49 | 23 | 0 | 0 |

Expert down-projection all-reduce (cannot change this block's routes; flips caused: 0): combined-output relative L2 vs oracle. No collective: 1.18e-06.

| n | bf16 ring | bf16 tree | fp32 ring | fp32 tree |
|---|---|---|---|---|
| 2 | 1.94e-03 | 1.94e-03 | 1.13e-06 | 1.13e-06 |
| 4 | 2.45e-03 | 2.31e-03 | 1.11e-06 | 1.11e-06 |
| 8 | 3.11e-03 | 2.64e-03 | 1.11e-06 | 1.10e-06 |
| 16 | 3.98e-03 | 2.86e-03 | 1.10e-06 | 1.10e-06 |
| 32 | 5.41e-03 | 3.09e-03 | 1.11e-06 | 1.10e-06 |
| 64 | 7.60e-03 | 3.31e-03 | 1.10e-06 | 1.10e-06 |

## Layer 7: route flips vs fp64 oracle from the router-input (o_proj) all-reduce
Tokens: 1348. No collective (n = 1): 0 flips.

| n | bf16 ring | bf16 tree | fp32 ring | fp32 tree |
|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0 |
| 4 | 4 | 3 | 0 | 0 |
| 8 | 3 | 4 | 0 | 0 |
| 16 | 8 | 3 | 0 | 0 |
| 32 | 8 | 2 | 0 | 0 |
| 64 | 8 | 5 | 0 | 0 |

Expert down-projection all-reduce (cannot change this block's routes; flips caused: 0): combined-output relative L2 vs oracle. No collective: 1.26e-06.

| n | bf16 ring | bf16 tree | fp32 ring | fp32 tree |
|---|---|---|---|---|
| 2 | 2.23e-03 | 2.23e-03 | 1.20e-06 | 1.20e-06 |
| 4 | 2.87e-03 | 2.76e-03 | 1.17e-06 | 1.17e-06 |
| 8 | 3.67e-03 | 3.19e-03 | 1.16e-06 | 1.16e-06 |
| 16 | 4.80e-03 | 3.57e-03 | 1.15e-06 | 1.15e-06 |
| 32 | 6.45e-03 | 3.91e-03 | 1.15e-06 | 1.15e-06 |
| 64 | 8.87e-03 | 4.22e-03 | 1.15e-06 | 1.15e-06 |

## Reading the numbers
- fp32 on the wire caused no route flips at any chip count and left output error at the no-collective level (~1.1e-6 relative L2).
- bf16 on the wire: route flips and output error grow with chip count; ring order grows faster than tree order (ring accumulates sequentially across chips, tree pairwise).
- Layer 0 is more sensitive than layer 7 (49 vs 8 flipped tokens at 64 chips, ring).
- p99 element ULP is in the parquet (`output_p99_ulp_vs_oracle`) but is dominated by near-zero outputs (the fp32 baseline alone is ~800-900 ULP at p99); relative L2 is the interpretable error measure.
- Single prompt set (32 prompts, 1,348 tokens), two layers, emulated reduction orders; no confidence intervals are attached to these counts.

Files: `h4_verdict.json`, `h4_collective_sweep.provenance.json` (copies); the per-configuration table `h4_collective_sweep.parquet` is in `data/results/h4-collective-sweep-fp32` (parquet files are gitignored).
