# Interfaces

Three file formats connect the components. Producers and consumers are built independently against this document, so
changes need the maintainer's sign-off and a note in the changelog at the bottom.

All arrays are NumPy `.npy` (little-endian, C order). Floating-point arrays are stored as `<f4` (float32) whose values
are exactly representable in the array's *logical dtype* (bf16 / fp16 / fp32), so widening is lossless; integer arrays are
`<i4`. Every directory carries a JSON file with provenance from `coinflip/provenance.py`.

Notation: T = tokens, H = hidden size (2048), I = expert intermediate size (1024), E = experts (64), K = top-k (8).

## 1. Layer export (producer: `reference/`; consumers: `port/`, `routediff/`)
One directory per (model, layer): `data/export/<model_slug>/layer<L>/`
```
config.json      model_id, hf_revision (commit sha), layer, hidden_size, intermediate_size, num_experts, top_k,
                 norm_topk_prob, rms_norm_eps, model_dtype ("bfloat16"), checkpoint_dtype,
                 transformers_version, experts_implementation ("eager"), torch_version, provenance
weights/
  o_proj.npy                    [H, H]       attention output projection (row-parallel under TP)
  post_attention_layernorm.npy  [H]          RMSNorm weight applied before the MoE block
  router.npy                    [E, H]       gate weight
  gate_up_proj.npy              [E, 2I, H]   rows 0..I-1 = gate, rows I..2I-1 = up (transformers chunk(2) order)
  down_proj.npy                 [E, H, I]
inputs/
  o_proj_in.npy                 [T, H]       attention output before o_proj
  residual.npy                  [T, H]       residual stream added to the o_proj output
  token_ids.npy                 [T]          int32, for traceability
  token_meta.json               per token: prompt id, position
```
The block computed by every backend is:
`router_input = RMSNorm(residual + o_proj_in @ o_proj^T)` then router -> softmax (fp32) -> top-k -> experts -> combine.

## 2. Per-op dumps (producers: `reference/`, `port/`; consumer: `routediff/`)
One directory per run: `data/dumps/<run_id>/` with `manifest.json` and one `.npy` per op.

`manifest.json`: `run_id`, `backend` (e.g. `torch-cpu`, `torch-mps`, `cpp`), `export` (path + config sha256),
`numerics` (backend settings, e.g. the C++ numerics JSON or torch dtype + experts implementation), `bug` (null, or the
bug spec's sha256 only, never its contents), `tokens` (T), `ops` (list below, in order, each with `file`, `shape`,
`logical_dtype`, `stage`), `provenance`.

| # | op | shape | logical dtype (reference) | stage |
|---|---|---|---|---|
| 1 | `o_proj_out` | [T, H] | model dtype | selection |
| 2 | `resid_sum` | [T, H] | model dtype | selection |
| 3 | `router_input` | [T, H] | model dtype | selection |
| 4 | `router_logits` | [T, E] | model dtype (gate output as stored) | selection |
| 5 | `router_probs` | [T, E] | fp32 | selection |
| 6 | `topk_ids` | [T, K] | int32, sorted by descending prob | selection |
| 7 | `topk_weights` | [T, K] | model dtype, aligned with `topk_ids` | post-selection |
| 8 | `expert_gate` | [T, K, I] | model dtype, slot-aligned | post-selection |
| 9 | `expert_up` | [T, K, I] | model dtype | post-selection |
| 10 | `expert_act` | [T, K, I] | model dtype, `silu(gate) * up` | post-selection |
| 11 | `expert_down` | [T, K, H] | model dtype | post-selection |
| 12 | `combine_terms` | [T, K, H] | model dtype, `expert_down * weight` | post-selection |
| 13 | `output` | [T, H] | model dtype | post-selection |

Rules: `topk_ids` are compared as sets per token. Slot-aligned expert ops are only compared on tokens whose top-k sets
match (on other tokens the difference is already explained by a selection op).

## 3. Running a backend with a bug or a patch (producer: `port/`; callers: `routediff/`, `evaluation/`)
```
port/build/moe_run --export DIR --numerics numerics.json --dump OUT_DIR \
                   [--bug bug.json] [--patch OP=path.npy ...] [--patch-ids topk_ids.npy] [--tokens a:b] [--threads N]
```
- `numerics.json`: `preset` (`hf_bf16` default, `hf_fp32`, `oracle`) followed by flat overrides: `storage`, `rms`,
  `gate_out`, `softmax`, `weights`, `combine_accum` (dtypes `fp64|fp32|bf16|fp16`); `{oproj,router,expert}_{accum,order,
  block,fma}` (orders `seq|pairwise|blocked|strided`); `topk_on` (`probs|logits`); `tiebreak` (`lower|higher`);
  `combine_order` (`expert_asc|expert_desc|slot`); `collective_{n,topology,wire,on}` (`ring|tree`; `none|oproj|down|both`).
  Unknown keys are an error. The resolved settings are echoed into the manifest.
- `--patch OP=path.npy`: replace that op's output with the given array, then recompute everything downstream. A patch
  on a slot-aligned op (`topk_weights` and the expert / combine ops) also needs `--patch-ids` (the patch source's
  `topk_ids`): values are matched per expert id, and only on tokens whose top-k set equals the source's; elsewhere the
  patch is skipped and `patch_skipped_token_ops` in the manifest counts it.
- Bug types and the op each one first changes: `port/src/block.cpp` `bug_registry()` (13 types).

**Bug spec** (`bug.json`, written by `evaluation/`, read only by `moe_run`):
`bug_id`, `type` (from the registry in `port/`), `params`, `injected_op` (the first op in the table above whose output
the bug changes), `seed`. routediff never reads bug specs or the directory they live in.

## 4. routediff output (producer: `routediff/`; consumer: `evaluation/`, write-up)
`report.json`: `reference_run`, `candidate_run`, per-op difference summary (max / p99 ULP, relative L2, tokens over
tolerance, route-set mismatches), `first_flagged_op`, `stage`, `patch_check` (op patched, criterion used: route set or
output tolerance, restored true/false, tokens restored / total), `tolerances` (file + sha256). Plus `report.md` for people.

## Changelog
- 2026-10-15: initial version.
- 2026-10-15: `moe_run` flags finalised (`--patch-ids`, `--threads`), numerics JSON keys listed, bug registry pointer.
