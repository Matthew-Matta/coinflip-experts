# Frozen-input prefill census

Use `--prepared-file PATH` to supply a successful gate summary (the default is the
recorded local gate); its immutable cached model revision is reused.

Corpus: 200 seeded prompts, 50 each GSM8K/HumanEval/ShareGPT/FLORES, prefix capped
at 256 tokens, followed by exactly 256 greedy reference continuation tokens.
Generation retains `output_ids` directly; no text re-tokenisation. Raw token IDs
stay under ignored `data/census/`; their SHA256 and manifest are committed here
**before measurement**. The runner verifies the committed HEAD marker.

```sh
# No GPU; inspect the launch plan first.
modal run modal/gate_olmoe.py --mode corpus
# Explicitly approved GPU spend only: generate reference continuations.
modal run modal/gate_olmoe.py --mode corpus --gpu-approved
# Matthew commits the SHA marker and manifest before the next command.
modal run modal/gate_olmoe.py --mode census --gpu-approved
```

SGLang 0.5.21 plus #40980 backport, same pinned gate image, bf16 L4 TP=1;
CUDA graphs and radix cache disabled; chunked-prefill-size=max-prefill-tokens=32768.
Patch source SHA256 is checked and recorded. Each request batch is <=32x512,
so the chunk capacity does not split these prefills. Four configurations:
bs1 twice (`r0a`, `r0b`), bs8 and bs32. `r0a` is also the bs1 comparison.
Every configuration uses identical frozen token IDs. The last bs32 batch has 8
sequences; this realised composition is recorded rather than hidden.

Only one HTTP batch is outstanding. Scheduler `Prefill batch` lines must show
exactly its sequence count, token sum and zero cached tokens. Since no other
requests run, this maps the submitted prompt IDs to that single realised batch;
the scheduler line itself reports counts rather than prompt identities. Any
split, missing log line or unexpected cached token stops the run. API route
arrays are decoded using `extract_routed_experts_from_meta_info` and shape/range
checked. Per-config JSON records versions, patch hash, timings and input SHA;
the parent summary records GPU/driver and measured memory. Results use
`coinflip.provenance.save_run`. Failures stop the server and preserve artifacts.

`python -m analysis.census --summary LOCAL_SUMMARY --out NEW_RESULT_JSON`
calculates paired excess over R0 and 10,000-resample prompt-cluster intervals.
Zero-event bootstrap intervals may be degenerate; do not interpret them as a
general population guarantee. These measurements cover **prefill only**.
Router margins and routing-weight flip mass are unavailable; no score-capture
instrumentation is included.

Source audit: pinned scheduler log evidence verified by the successful gate.
Current checkout emits the scheduler fields in
`third_party/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py:710`;
route-buffer sizing is in `state_capturer/routed_experts.py:79`. The census keeps
the gate backport and capacity workaround rather than modifying those kernels.
