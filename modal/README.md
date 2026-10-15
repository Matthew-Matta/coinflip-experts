# L4 gate

```sh
python modal/gate_olmoe.py --dry-run
modal run modal/gate_olmoe.py
```

The second command builds the pinned image and runs only a CPU download into the
`coinflip-models` Volume, using the `huggingface` Modal secret. CPU preparation has
costs but does not allocate a GPU. The model revision is resolved to an immutable
SHA. The GPU reads that local snapshot in offline mode.

After explicit approval of GPU spending:

```sh
modal run modal/gate_olmoe.py --gpu-approved
```

One L4 job has a 30-minute hard timeout, no automatic retries, a ten-minute server
health deadline and 120-second HTTP deadlines. It checks the driver, package pins,
model dimensions, greedy output, route capture, repeated prefill route sets and
512-token requested batches of 1, 8 and 32. It terminates the server process group
and writes a summary, server log and sampled device-memory usage to the Volume.
Memory is sampled via `nvidia-smi` because inference runs in a child process;
PyTorch peak counters in the parent would miss it. Sampled peaks can miss short spikes.

These are compatibility checks, not census data. HTTP batch size is labelled
requested, not proven realised scheduler batch size. Prefix caching is disabled,
as are CUDA graphs and chunked prefill. Realised scheduler composition needs its
own validation before the census.

Source audit: SGLang v0.5.21, commit
`e00930c5489053f26d86b179cee0d087f846acbb`:

- `python/sglang/srt/state_capturer/routed_experts.py:153-161`: helper decodes
  `meta_info.routed_experts` into int32.
- `python/sglang/srt/state_capturer/base.py:173-188`: returned positions cover
  `[start_len, seqlen-1)`.
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:150-163`:
  sequence length is input plus output tokens. With exactly one output token,
  the returned rows cover the complete prefill and no decode position.
