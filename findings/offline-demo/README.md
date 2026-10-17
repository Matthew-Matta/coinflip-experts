# Offline one-command demo

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

**Verified locally** in an isolated committed-source snapshot plus the new demo
files, a fresh virtualenv with only base/dev dependencies, no Torch installed,
and no private model exports. C++ was rebuilt serially in that temporary source
tree, leaving the active sweep's build alone. `verification.json` records the
result. A dedicated Ubuntu CI job repeats installation with only `.[dev]`, the
C++ build and `make demo`; that new CI job has not been run by this task.
