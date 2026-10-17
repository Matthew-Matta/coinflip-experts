# MPS pre-routing hook demo (not blinded)

Real OLMoE layer 0, bf16 eager, Apple M1 Pro MPS, 64 captured positions
(first two valid positions from each of 32 original prompts). CPU threads: 2;
MPS CPU fallback disabled. Clean mirror equals the actual Transformers module
bit-exactly. Rerun after the fused-layout correction; CPU demo retained unchanged.

A known forward hook multiplies **router_logits by -1 before softmax/top-k**.
This reverses expert preference and changes **64/64 route sets**. Routediff
localises **router_logits**, patches it and recomputes downstream via the torch
adapter. The route-set criterion restores the clean reference. An independent
repaired rerun is **bit-exact at all 13/13 operations**.

The unchanged bf16-exploratory tolerance SHA is recorded in `summary.json`.
This conspicuous fault demonstrates hook localisation and causal repair; it is
not blinded evaluation or evidence about sensitivity to small numerical faults.

```sh
python -m reference.torch_hook_demo --device mps --fault router --out data/results/new-mps-router-demo
```
