# Cross-chip all-reduce: bf16 on the wire flips routes, fp32 does not

**In one line:** summing the router input across emulated chips in fp32 flips **0** routes at every chip count;
in bf16 it flips **10 to 49 of 1348** tokens on layer 0 as chips go from 2 to 64.
- **What I found:** flips grow with chip count for a bf16 ring (Spearman rho **0.899** layer 0, **0.880** layer 7);
  the preregistered test H4 passes on both layers. Output error from a bf16 down-projection all-reduce grows too.
- **Why it matters:** a multi-chip port should all-reduce the router input in fp32 (numerics contract R1).
- **Caveat:** emulation of stated summation orders on one CPU, not a measurement of any interconnect; small counts.

**Type:** preregistered (H4)   ·   **Reproduce:** `python -m port.sweeps.collective_sweep --export-root data/export/olmoe-1b-7b-0924 --layers 0 7 --base hf_fp32 --out NEW_DIR`

---

## Details

**Emulation** (PREREG section 3), not a measurement of any interconnect. Base numerics `hf_fp32` (PREREG amendment
c18d512), compared with the fp64 oracle. Margin-free: H4 has its own preregistered criteria.

- Binary: `port/build/moe_run` sha256 `c30b6f2faee45cb23ada7d8237bd5ca544a5bb59c8d1ed2d7fb3dac420c55a1b` (unchanged
  since 18:13:58; matches an independent recheck of the hash). Started 10-16 18:53:35 at commit 5dc67ef, PREREG
  amendment c18d512 committed 18:35:22; wall time 80 min.
- Provenance git state is captured at save time (`cd74858`, dirty: uncommitted SIMD source edits that did not affect
  this binary).

### Verdict (PREREG H4)
- layer0: **PASS**. (a) bf16 ring, Spearman rho over n = 2..64: 0.899 (>= 0.5). (b) fp32 wire: ring pass (rule of three, bound 0.0022, max rate 0.0000); tree pass (rule of three, bound 0.0022, max rate 0.0000)
- layer7: **PASS**. (a) bf16 ring, Spearman rho over n = 2..64: 0.880 (>= 0.5). (b) fp32 wire: ring pass (rule of three, bound 0.0022, max rate 0.0000); tree pass (rule of three, bound 0.0022, max rate 0.0000)

### Layer 0: route flips vs fp64 oracle from the router-input (o_proj) all-reduce
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

### Layer 7: route flips vs fp64 oracle from the router-input (o_proj) all-reduce
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

### Reading the numbers
- fp32 on the wire caused no route flips at any chip count and left output error at the no-collective level (~1.1e-6 relative L2).
- bf16 on the wire: route flips and output error grow with chip count; ring order grows faster than tree order (ring accumulates sequentially across chips, tree pairwise).
- Layer 0 is more sensitive than layer 7 (49 vs 8 flipped tokens at 64 chips, ring).
- p99 element ULP is in the parquet (`output_p99_ulp_vs_oracle`) but is dominated by near-zero outputs (the fp32 baseline alone is ~800-900 ULP at p99); relative L2 is the interpretable error measure.
- Single prompt set (32 prompts, 1,348 tokens), two layers, emulated reduction orders; prompt-cluster confidence intervals are now provided in [bootstrap/README.md](bootstrap/README.md), with a modelled accuracy-vs-bytes chart.

Files: `h4_verdict.json` (copy; `h4_collective_sweep.provenance.json` is next to it locally but `*.provenance.json` is gitignored); the per-configuration table `h4_collective_sweep.parquet` is in `data/results/h4-collective-sweep-fp32` (parquet files are gitignored).
