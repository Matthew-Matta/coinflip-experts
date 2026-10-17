# Case study: routediff on the real M2 discrepancy (post hoc, descriptive)

Label: post-hoc case study, routediff applied to a real discrepancy found during M2. Not part of the blinded evaluation.

## Setup
- Reference: `data/dumps/layer0-torch-float32` (PyTorch CPU fp32, transformers 5.12.1 eager).
- Candidate: the frozen pre-fix C++ dump `data/results/m2-initial-layer0/dumps/layer0-cpp-hf_fp32` (read-only).
- Reruns: the pre-fix port rebuilt from `267ddfb` (HEAD at freeze) at `data/results/case-study-rmsnorm/port-267ddfb/` (local), binary sha256 8891bf7a2b17446082e81efc9fb4734f48566295022f4ece8b302fb4404b2285; it reproduces the frozen pre-fix dump 13/13 ops bit-identically (`data/results/case-study-rmsnorm/repro-prefix-fp32`). Numerics: the frozen `configs/numerics_hf_fp32.json`.
- Split: evaluation (664 tokens).
- Tolerances: `findings/routediff-eval-real-v2/tolerances.json` (published byte-identical copy of `data/evaluation/real-v2/tolerances.json`), sha256 1944f35dfec6c5e3ff108f482cc47c8effb7a20e8c1d820d2d5c92d4cfd886ff (Codex, frozen; margin 2.0 per PREREG amendment 5dc67ef).

## Result 1 (as specified, frozen tolerances): nothing flagged
`report-realv2-tol.{json,md}`: **no op exceeds its tolerance**; 0/664 route-set mismatches. The pre-fix RMSNorm order difference (worst 17 scaled ULP at `router_input`, p99 element 9 ULP) is far inside the benign envelope (tolerance 366), which is set by permitted numerics choices (pairwise vs sequential, FMA vs not). In calibrated mode routediff treats it as within what a correct port may do, consistent with it changing no expert choices. The M2 discrepancy was found by the strict H-port fidelity test, not by this mode.

## Result 2 (supplementary, strict mode: no tolerances, any difference counts)
`report-strict-supplementary.{json,md}`: first flagged op `router_input`; cumulative patch-forward localises to **`router_input`**, with sites `router_input` (RMSNorm reduction order), `router_probs` (softmax exp) and `expert_act` (SiLU exp). Patching those three restores the output bit-exactly; `expert_down` is not a site (its difference is inherited from `expert_act`). This reproduces the M2 manual diagnosis automatically.

## Takeaway
Calibrated tolerances answer "is this outside permitted numerics?"; strict mode answers "where does the port first deviate from the reference?". The RMSNorm order mismatch is the second kind only.
