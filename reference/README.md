# OLMoE block reference

Install with `pip install -e '.[dev,reference,modal]'`. Model exports, prompts and
per-operation dumps go under ignored `data/`; no weights or prompt text belong in Git.

```sh
python -m reference.select_prompts
python -m reference.export_block --pool data/prompts/block_pool.jsonl
python -m reference.dump_block --export data/export/olmoe-1b-7b-0924/layer0 --dtype bfloat16 --dump data/dumps/layer0-torch-bf16
python -m reference.dump_block --export data/export/olmoe-1b-7b-0924/layer0 --dtype float32 --dump data/dumps/layer0-torch-fp32
```

The prompt pool contains up to 128 eligible rows per source. HF dataset revisions
are pinned; FLORES uses Meta's original public archive with its SHA-256 recorded.
The exporter selects eight per source with a fixed seed, assigns four to calibration
and four to evaluation, and truncates each input to 64 tokens. Selection, source IDs,
token IDs and splits are saved before inference. This is the block-reference input
set, separate from the future serving census.

The exporter loads `OlmoeModel` (no language-model head), truncated after the deepest
requested layer that fits the weight budget. Default budget: 8 GiB of bf16 weights;
remaining RAM is reserved for Python, tensors, inference and widened float32 exports.
It prints skipped layers, wall time and process peak RSS. That cap is an estimate,
not an OS memory limit. Never increase it on a 16 GB machine to force layer 15 to fit;
use a high-memory CPU job instead. Outputs are never silently overwritten.

## Reference semantics

Pinned `transformers==5.12.1`, installed
`transformers/models/olmoe/modeling_olmoe.py`:

- Lines 58-63: RMSNorm computes variance in fp32, casts the normalised result back
  before multiplying by the norm weight.
- Lines 310-311: fused expert gate/up weights `[E, 2I, H]`, down `[E, H, I]`.
- Lines 321-336: ascending hit-expert order; slot-major token indexing; fused
  gate/up projection followed by `chunk(2)`; activation, down projection, routing
  weight multiplication, then model-dtype `index_add_`.
- Lines 352-357: model-dtype gate logits, fp32 softmax, top-k on probabilities,
  optional renormalisation, routing weights cast to the logits dtype.
- Lines 397-418: layer-entry residual plus attention output, then post-attention
  RMSNorm. Pre-hooks capture the layer-entry residual and attention `o_proj` input.

The dump mirror uses these operations and refuses to write a successful manifest
unless its combined output equals the actual Transformers eager module bit-exactly.
It does not claim C++ fidelity: the C++ comparison is a separate validation. The
export and dump manifests record versions, source SHA-256 and runtime provenance.
PyTorch exact-tie ordering is retained rather than replaced by a custom tie rule.

## Other real backends

`python -m reference.run_pairs --help` generates CPU grouped_mm and MPS eager
variants in both dtypes. Each is validated against the actual selected module;
MPS never silently falls back to CPU. See `analysis/README.md` for comparisons.
