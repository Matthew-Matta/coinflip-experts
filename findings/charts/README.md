# README chart inventory

These paths are relative to the repository root. The two available charts were
visually checked locally; their numerical sources remain separate experiments.

| Chart | PNG (SVG alongside it) | Status |
| --- | --- | --- |
| Census per-layer route-set flip rates | `findings/census-olmoe-prefill/per_layer_flips.png` | Available; descriptive prefill results, bs8 and bs32 versus bs1 |
| H4 accuracy versus modelled bytes | `findings/h4-collective-sweep-fp32/bootstrap/accuracy_vs_bytes.png` | Available; numerical emulation plus communication model, prompt-cluster 95% CIs |
| Accumulator-width flips versus fraction bits | `findings/accumulator-width/readme-chart/flips_vs_bits.png` | Pending completion of the coder's sweep; not generated from partial rows |

The census is not compared numerically with R3. The H4 chart is not an
interconnect measurement.

Once the accumulator sweep has written its final parquet, provenance and summary:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 MPLCONFIGDIR=/tmp/coinflip-mpl python -m analysis.accumulator_chart --source data/results/accumulator-width --out findings/accumulator-width/readme-chart
```

The generator refuses incomplete grids, retains zero flip rates on its axis,
keeps layers separate, and records source hashes. Its chart must be visually
checked after generation. The output directory must not already exist.
