# SGLang OLMoE prefill census: H1 PASS

**Batch-size-associated captured route flips are nonzero on this setup.**
These are prefill-only expert-set comparisons on fixed token IDs, not decode
or op-level attribution. No router scores/margins were captured.

| Configuration vs bs1 | Flipped token-layer sets / total | Rate (95% prompt-cluster CI) | Sequences with >=1 flip |
| --- | --- | --- | --- |
| R0: bs1 repeated | 0 / 1,059,312 | 0% [0%, 0%] | 0/200 |
| bs8 | 76,852 / 1,059,312 | 7.255% [7.026%, 7.484%] | 200/200 |
| bs32 | 76,371 / 1,059,312 | 7.209% [6.978%, 7.440%] | 200/200 |

R0 is zero, so excess-over-R0 rates and intervals equal the displayed rates.
H1 passes because the bs32 excess 95% CI excludes zero. The <0.01% upper-bound
kill check is not met. Zero/all-success bootstrap intervals for R0 or sequence
fractions are degenerate and do not establish population guarantees.

## Frozen inputs and configuration

- 200 prompts, 50 each GSM8K/HumanEval/ShareGPT/FLORES; seeded selection from
  the pinned source pool. Prompt prefixes <=256 tokens; exactly 256 greedy
  reference continuation tokens retained verbatim as output IDs; <=512 total.
  66,207 input positions x 16 MoE layers = 1,059,312 paired token-layer units.
- Token JSON SHA256
  `2cebf9203960a7e53756a9c82d23c98de2e560bd42dcc190bc2e5a28bf2bfb54`
  was committed in **446a873 before measurement**. The runner checks it against
  HEAD before launching. [Committed marker](../../census/olmoe_tokens_v1.sha256).
- OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`; bf16, one L4, TP=1;
  SGLang 0.5.21 + backport of #40980. Patched `sglang/srt/models/olmoe.py` SHA256:
  `4de433fa6dbceafec19fb10ad44d05120135aba8f6629555f119f75260076ee4`.
- CUDA graphs and radix cache disabled; chunked-prefill-size and
  max-prefill-tokens both 32768. Every observed batch ran as one unchunked
  prefill. This is not a default-production-server benchmark.
- Selection seed42, batch seed20261017. One seeded prompt permutation is reused
  across configs: bs1 twice (200 batches each), bs8 (25 batches), bs32 (six
  32-sequence batches + one 8-sequence batch). The bs1 baseline doubles as R0a.
- All **432 measurement scheduler “Prefill batch” lines** were parsed and independently
  checked against submitted prompt IDs, sequence counts, frozen input-token
  sums and zero cached tokens. Only one request batch was outstanding at a time;
  scheduler counts plus that invariant identify its realised membership. Two
  startup/health prefills (6 tokens and 1 token) precede the first successful
  health response and are excluded from the measurement, as the runner did.
- GPU phase wall time including startup: **272.4 seconds**. Sampled peak device
  memory: **21,859 MiB (21.35 GiB)** at 0.2-second polling; not an exact
  instantaneous maximum. Driver/library versions and per-run patch hash, input
  hash, timings and GPU data are retained in the raw per-config `run.json` files.

## Statistics and verification

Paired top-k **sets**, not ordered lists; token-length-weighted rates. Percentile
95% CIs use **10,000 prompt-cluster resamples**, seed20261017, preserving all
positions/layers within each prompt. Layers/dtypes/GPU setups from other
experiments are not pooled here. Only one model revision, GPU and neighbour
permutation were tested; the result does not identify which numerical operation
causes a flip or quantify output quality impact.

A separate CPU pass re-read every saved route array, recomputed all flip counts,
verified complete per-config prompt coverage, and checked patch/token hashes.
The downloaded archive SHA256 matches `verification.json`. Local checks then
compared raw scheduler lines and every realised batch against the original
frozen token file. GPU inference was not rerun for this verification.

`summary.json` contains rates/CIs; `metadata.json` is the run metadata with its
hostname redacted and clearly labelled supplementary source-derived settings;
local RPC and remote summary checksums are both recorded (their JSON contents
match; their trailing-newline formatting differs);
`verification.json` records independent checks. Raw parquet plus provenance
(`coinflip.provenance.save_run`), arrays, scheduler log and per-run settings are
in ignored `data/census/downloaded/census-6549ef998bce/`. The frozen token file
is `data/census/olmoe_tokens_v1.json`. SHA-verified measurement-time Python source
snapshots are included in `source/`; the normal runner later gained archive and
successful-gate-input checks without changing this measurement.

## Reproduce

```sh
# Explicit spend approval and the committed token hash are required.
modal run modal/gate_olmoe.py --mode census
modal run modal/gate_olmoe.py --mode census --gpu-approved
python -m analysis.census --summary NEW_RUN/summary.json --out NEW_RUN/analysis.json
```

The optional capture-buffer upstream PR has not been opened. This experiment
uses the declared capacity workaround, not a new capture-buffer implementation.

## Descriptive structure of the observed differences

![Captured route flips by layer](per_layer_flips.png)

For **bs8 vs bs1**, the layer-0 rate is **2.619%** (1,734/66,207), rising
non-monotonically to **10.555%** at layer 14 (6,988/66,207), then 9.599% at
layer 15. The chart also shows bs32 separately; it has a similar observed profile.

Of the 76,852 differing bs8 token-layer sets, **75,208 (97.861%) swap exactly
one expert**: one expert leaves and one enters the top-8 set. This is not two
independent flips. Larger replacements account for the remaining 1,644 sets.
**34,655/66,207 token positions (52.343%)** differ in at least one layer;
**zero** differ in all 16 layers. These are token-position rates, distinct from
the earlier 200/200 sequence-level result. For bs32, the corresponding figures
are 74,696/76,371 one-expert swaps (97.807%), 34,454/66,207 positions with any
layer flip (52.040%), and zero positions flipping in all layers.

These are descriptive counts on the frozen corpus, not an attribution to a
specific numerical operation or an explanation of downstream quality.
`structure.json` includes the full replacement-size and flipped-layer-count
histograms plus input hashes; `per_layer.csv` contains all 16 rates. Reproduce:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python -m analysis.census_structure --source data/census/downloaded/census-6549ef998bce --out NEW_OUTPUT_DIRECTORY
```
