# Qwen3-30B-A3B layer-0 reference export

**PASS:** a one-layer truncated model, eager attention and experts pinned,
produces bf16 and fp32 MoE-block mirrors bit-exact to the Transformers module.
This is a layer reference check, not a full-model serving experiment.

- Checkpoint revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- The safetensors index selects only `model-00001-of-00016.safetensors`
  for embedding plus all layer-0 tensors (3,999,417,504 bytes downloaded).
  No other checkpoint shard is downloaded or loaded.
- Same 32 original prompt IDs/text and calibration/evaluation split; Qwen's
  tokenizer gives 1,361 valid positions (OLMoE used 1,348). Inputs are paired
  within Qwen, not token-identical across tokenizers/models.
- Peak process RSS **3.660 GiB**, including loading, capture, export and dumps.
- Final model norm is omitted because it is downstream of all captured
  operations; no logits, generation or full-model equivalence is claimed.
- Each of 13 ops is dumped with the existing lossless float32 storage convention;
  checkpoint/model arithmetic is bf16 during capture, then the MoE block is run
  in bf16 and fp32 separately with `mirror == module` assertions.

Qwen has a **rectangular o_proj**. The export records `o_proj_input_size` and
`port_compatible=False`; the reference reader handles that input width. The
C++ port at the time of this export could not consume it. That historical
metadata is retained unchanged. Rectangular o_proj support was subsequently
added in 43ee582; later parity results are in [qwen3-port](../qwen3-port/).
No shape was coerced during this export.

```sh
HF_HOME=data/hf-cache python -m reference.export_qwen --out NEW_OUTPUT_DIRECTORY
```

Raw arrays/dumps/selection remain under ignored
`data/export/qwen3-30b-a3b-layer0`. `summary.json` records RSS and timings;
`download_plan.json` records the index hash and exact selected tensor keys.
Installed source: transformers 5.12.1 `models/qwen3_moe/modeling_qwen3_moe.py`,
expert loop lines 214–251, router lines 254–272, sparse block lines 275–289.
