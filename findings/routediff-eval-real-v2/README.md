# Blinded routediff evaluation: real-v2

**H-routediff PASS** under the amended, frozen margin-2.0 calibration
(PREREG amendment `5dc67ef`). Reference: PyTorch CPU fp32; candidate: C++
`hf_fp32`; OLMoE layers 0 and 7. Inputs were previously inspected during M2;
the evaluation prompts were held out from tolerance calibration, not unseen.

| Metric | Result |
| --- | --- |
| Detection, effective bugs | 44/44 (100%) |
| Correct first-site localisation, detected bugs | 44/44 (100%) |
| False positives, benign controls | 0/32 (0%) |
| No-effect injected bugs | 8 (zero alarms) |
| Execution errors | 0 |

All 84 reports were sealed before scoring. The 8 no-effect injections are four
combine-order changes within benign tolerance and four tie-break reversals with
no fp32 ties; they are excluded from the effective-bug denominator.

## Limits on interpretation

- The preregistered prompt-cluster bootstrap (10,000 resamples, 16 evaluation
  prompts) is degenerate: detection/localisation [1, 1], FP [0, 0]. Two-sided
  95% Clopper–Pearson case-level intervals instead give detection/localisation
  lower bound about **0.920** and FP upper bound about **0.109**. These assume
  independent cases, whereas cases share prompts, and are supplementary rather
  than a replacement for cluster inference. The point estimates pass the
  thresholds; the sample does not establish population FP <=5%.
- The verifier found injected dtype-bug effects roughly **250x–1,000,000x** the
  calibrated tolerance. This validates debugging mechanics on large faults,
  **not sensitivity to subtle faults**; this range is not claimed for every bug.
- Route-set restoration was the criterion in **11/44** cases; **33/44**
  localisations restored the output tolerance. Thus the headline localisation
  rate primarily validates output debugging, not route-flip attribution alone.
- Repeats use different seeded prompts, not independent implementations of a bug.
  Each of 13 registry bug types has two trials per layer.

## Margin-1.0 sensitivity: partial real-v1

Only the first **15 completed cases** were scored after real-v2 was sealed,
as required by amendment `5dc67ef`: detection **13/13**, localisation **13/13**,
plus two no-effect injections with no alarms. **No benign cases completed**,
so there is **no false-positive evidence at margin 1.0**, not a measured 0% FP
rate. This partial prefix is not a complete evaluation or an independent sample.

## Artifacts and provenance

`summary.json` and `frozen.json` are exact copies of the sealed-run artifacts.
`settings.json` is a public copy: the repository's absolute path is replaced by
`<repo>` and the hostname is redacted; source hashes and numerical settings are
unchanged. Consequently the original settings hash in `frozen.json` refers to
the private original, not the redacted copy. No bug specifications are included.

Binary built from committed HEAD `9dd291de4a84f21f6c468f0f66f218ce72298f8f`
in an isolated worktree; binary SHA256:
`c30b6f2faee45cb23ada7d8237bd5ca544a5bb59c8d1ed2d7fb3dac420c55a1b`.
Seed 20261016, two repeats, margin 2.0. Raw runs remain in ignored
`data/evaluation/real-v2`; partial sensitivity scores in
`data/evaluation/real-v1-scoring`. Source runs were not modified.
