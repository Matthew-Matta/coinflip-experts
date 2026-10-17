# Patching real Torch backends

**In one line:** All four backend pairs stayed within tolerance, so I ran **zero** fault-localisation patches.

- **What I found:** I checked that replacing each operation with its own values preserves the real backend results.
- **Why it matters:** The adapter supports controlled repairs on CPU and Apple GPU paths.
- **Caveat:** These clean comparisons do not prove causal repair of a real detected bug.

**Type:** exploratory   ·   **Reproduce:** `python -m reference.backend_pairs --export data/export/olmoe-1b-7b-0924/layer0 --dtype bfloat16 --device mps --implementation eager --dump NEW_OUTPUT_DIRECTORY`

---

## Details

I use `reference.backend_pairs` to rerun a backend with replacement values:
`--out {out} {patch}`, including repeated `--patch OP=FILE` and `--patch-ids`.
Forward hooks on Identity taps at the 13 named stages replace values in the
validated numerical mirror; downstream operations are recomputed on MPS eager
or CPU grouped_mm. This instruments a mirror, not Transformers' internal
functional calls. With no patches, the result is asserted bit-exact to the
actual pinned Transformers module. Patched runs do not claim that unmodified
module equality.

**Validation:** all 13 unpatched intermediate arrays still match the original
real dumps bit-exactly for layers 0 and 7, MPS eager and CPU grouped_mm bf16.
Patching each stage with its own unpatched value leaves all 13 ops bit-exact on
all four pairs. CPU unit tests cover bf16/fp32 eager/grouped, downstream
recomputation, and expert-ID slot permutation. Slot-aligned patches are only
applied where route sets match and are remapped by expert ID; skipped token-op
counts are recorded.

Full routediff comparison ran on the held-out evaluation split (664 tokens per
layer) with `findings/bf16-tolerances/tolerances-bf16-exploratory.json`
and the torch rerun command enabled. **All four pairs are within tolerance**;
therefore the engine requested **zero patch checks** and localised no fault.
This is not causal patch-forward evidence on a detected real bug. MPS route-set
mismatches are 6 (L0) and 4 (L7), inside the calibrated exact-tie allowance;
grouped_mm has zero. The bf16 tolerance recipe is exploratory and its output
tolerance is inflated by tie mismatches (see the bf16-tolerance findings).

I kept the raw reports in `data/results/torch-patch-forward-bf16`; `summary.json` and
per-pair null-patch verification records are copied here. No planted ground
truth, no H-routediff rescore, and no alteration of frozen tolerances.
