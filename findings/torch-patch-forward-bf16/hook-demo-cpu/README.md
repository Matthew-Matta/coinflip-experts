# Repairing a known CPU fault

**In one line:** Patching the injected fault restored all **13** intermediate arrays exactly.

- **What I found:** I multiplied expert contributions before combining them, then localised and repaired that operation.
- **Why it matters:** This demonstrates a repair through the Torch backend adapter.
- **Caveat:** The fault is conspicuous and known in advance; this is not blinded or sensitivity evidence.

**Type:** exploratory   ·   **Reproduce:** `python -m reference.torch_hook_demo --device cpu --out NEW_OUTPUT_DIRECTORY`

---

## Details

**CPU only.** I ran this demonstration on CPU without a new MPS run.
It is a CPU substitute, not MPS evidence.

On real OLMoE layer-0 bf16 data (64 positions, two per original prompt), a forward
hook multiplies `combine_terms` by 16. The repaired-value hook executes after the
fault hook, so replacement can remove the fault. No fault label is scored as a
blinded result, and the multiplier is a conspicuous mechanics demonstration,
not a sensitivity test.

With the existing exploratory bf16 tolerances (SHA256 in `summary.json`),
routediff flags and **localises `combine_terms`**, then runs the full rerun
contract and confirms output restoration. The route sets remain unchanged,
as expected for a post-selection fault. A separate repaired rerun confirms
**all 13 op arrays are bit-exact to the clean run**, stronger than merely
returning below the output tolerance. The clean mirror is checked against the
actual Transformers module before fault injection.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python -m reference.torch_hook_demo --device cpu --out NEW_OUTPUT_DIRECTORY
```

Each process uses at most two numerical CPU threads. `report.json`/`report.md`
are the actual localiser results, with the machine-specific repository prefix
redacted. Raw arrays are in `data/results/torch-hook-demo-cpu`.
