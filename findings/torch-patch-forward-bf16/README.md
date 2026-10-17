# Torch hook adapter and real bf16 pairs (exploratory)

`reference.backend_pairs` accepts the routediff rerun contract:
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
and the torch rerun command enabled. **All four pairs are inside the envelope**;
therefore the engine requested **zero patch checks** and localised no fault.
This is not causal patch-forward evidence on a detected real bug. MPS route-set
mismatches are 6 (L0) and 4 (L7), inside the calibrated exact-tie allowance;
grouped_mm has zero. The bf16 tolerance recipe is exploratory and its output
tolerance is inflated by tie mismatches (see the bf16-tolerance findings).

Raw reports are in `data/results/torch-patch-forward-bf16`; `summary.json` and
per-pair null-patch verification records are copied here. No planted ground
truth, no H-routediff rescore, and no alteration of frozen tolerances.
