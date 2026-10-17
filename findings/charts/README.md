# README chart inventory

These paths are relative to the repository root. The three available charts were
visually checked locally; their numerical sources remain separate experiments.

| Chart | PNG (SVG alongside it) | Status |
| --- | --- | --- |
| Census per-layer route-set flip rates | `findings/census-olmoe-prefill/per_layer_flips.png` | Available; descriptive prefill results, bs8 and bs32 versus bs1 |
| H4 accuracy versus modelled bytes | `findings/h4-collective-sweep-fp32/bootstrap/accuracy_vs_bytes.png` | Available; numerical emulation plus communication model, prompt-cluster 95% CIs |
| Accumulator-width flips versus fraction bits | `findings/accumulator-width/flips_vs_bits.png` | Available; exploratory completed sweep |

The census is not compared numerically with R3. The H4 chart is not an
interconnect measurement.

The completed accumulator chart uses the saved sweep parquet:

```sh
python port/sweeps/plot_accumulator_width.py data/results/accumulator-width/accumulator_width.parquet --out NEW_OUTPUT_DIRECTORY/flips_vs_bits.png
```

See [the accumulator findings](../accumulator-width/README.md) for interpretation.
