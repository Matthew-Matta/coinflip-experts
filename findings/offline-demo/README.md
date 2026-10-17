# Try the debugger offline

**In one line:** I localise and repair **one** synthetic router fault without a model download.

- **What I found:** I built a seeded example that checks both the suspect operation and the repaired output.
- **Why it matters:** A reader can try the complete debugging path locally.
- **Caveat:** This is a mechanics demo, not evidence about real-model accuracy or subtle faults.

**Type:** descriptive   ·   **Reproduce:** `make demo`

---

## Details

```sh
pip install -e ".[dev]"
cmake -S port -B port/build
cmake --build port/build --target moe_run --parallel 2
make demo
```

`make demo` requires no Torch, model download, GPU, network, private export or
pre-existing data. It creates a tiny seeded synthetic export, compares C++ fp32
against the same port with bf16 router-logit output, invokes routediff's full
patch-forward CLI, and asserts localisation to **router_logits** with successful
restoration. It prints the localised op and Markdown report path. Each invocation
writes a new timestamped directory under `data/demo/`.

This is a synthetic mechanics demo using strict comparison, not a real-model
measurement, calibrated detection-power claim or blinded evaluation.

I verified this in an isolated committed-source snapshot plus the demo
files, a fresh virtualenv with only base/dev dependencies, no Torch installed,
and no private model exports. I rebuilt C++ serially in that temporary source
tree. `verification.json` records the
result. A dedicated Ubuntu CI job repeats installation with only `.[dev]`, the
C++ build and `make demo`; I had not run that new CI job when I recorded this result.
