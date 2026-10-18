# Site data contract (version 1)

Static JSON, UTF-8; total payload below 300,000 bytes. Expert IDs and token indices are zero-based.
Rates and shares are fractions (multiply by 100 for percent); counts are integers. No experiments run during export.
Every file has `schema_version: 1` and `provenance: {sources: [{path: string, sha256: string}]}`.
The exporter source is also hashed in every provenance block. Paths are repository-relative; hashes cover original bytes, including local raw dumps needed for regeneration.
No timestamps or machine-specific absolute paths are generated. Missing measurements are null, never zero.

## pipeline.json

`ops: Op[]` in execution order; `edges: [string,string][]` are data dependencies.
Op fields: `id`, `display_name`, `explanation` (strings), `shape` (integer array, one token),
`dtype` (string, primary fp32 configuration), `pytorch` (string, transformers 5.12.1 modeling_olmoe.py:line),
`cpp` (string, function in port/src/block.cpp). Shape/dtype: data/dumps/layer0-torch-float32/manifest.json.
Descriptions and dependencies mirror reference/dump_block.py, installed transformers source and port/src/block.cpp.

## scenarios.json

`token: {layer: integer, index: integer, prompt_id: string, position: integer, split: string, normalized_margin: number, probabilities: number[64], top8: integer[8]}`.
Token identity comes from data/export/olmoe-1b-7b-0924/layer0/inputs/token_meta.json.
Select the smallest positive (p8-p9)/p8 in the fp32 layer-0 raw dump among C++ route-matched rows;
ties in the selection resolve to the first token index. Source: data/dumps/layer0-torch-float32 and
data/results/simd/layer0-cpp-hf_fp32-scalar (router_probs.npy, topk_ids.npy). `probabilities` are unrenormalized router softmax probabilities.

`scenarios: Scenario[]` has clean, planted_bug, apple_gpu_fault. Each has `id`, `label`, `bug_type`
(strings; clean bug_type is null), `batch_scope` (string explaining batch coverage), `source_report` (path string), `token_index` (integer in that scenario's batch),
`before` and `after` (State; clean after is null). The global near-tie token is used for clean; other scenarios use
one token from their own captured batch, explicitly indexed, rather than pretending the batches are identical.
State: `batch` (original report object), `source_runs` ({reference, candidate: path strings}), `token` (TokenState), `routes` ({reference: integer[8], candidate: integer[8]}).
TokenState: `ops` maps each op ID to `{max_scaled_ulp: number|null, tolerance_scaled_ulp: number|null,
compared: boolean, flagged: boolean}`; `route_sets_equal` boolean. Scaled ULP is absolute difference divided by
reference spacing at max(abs(reference), row RMS), maximized over the token's elements. Slot ops compare only
when route sets match, aligned by expert ID. topk_ids uses set comparison and null numeric ULP/tolerance; it is flagged only when a different choice remains a valid top-k under the candidate probabilities (an unexplained selection difference).
Batch reports retain their source fields, including ops, route_mismatch_tokens, first_flagged_op and patch_checks;
null numeric values mean unmeasured, and compared_tokens=0 must never be presented as an error-free comparison.
Before reports: sealed data/evaluation/real-v2/cases/*/report/report.json (router_accum), frozen benign
real-v2 report (clean), findings/torch-patch-forward-bf16/hook-demo-mps-router/report.json (Apple).
The clean batch is a separate sealed benign evaluation batch, identified by its source report; its token view uses
the full scalar C++ dump, identified by source_runs. Do not imply that the global token belongs to that batch.
After states use existing repaired dumps for the Apple demo; its batch object is the saved restoration summary.
The original sealed evaluator deleted its temporary patched dumps: planted_bug.after instead has
`batch: {patch_check: object}` copied verbatim, `source_runs`, `token`, `routes` all null and `unavailable: string`.
No after-patch ULP values or token routes are invented from the restoration boolean. No batch statistics are recomputed.

`exact_tie`: {layer, index (integers), probabilities (number[2]), tied_experts (integer[]),
choices: {cpp, apple_gpu, pytorch_macos, pytorch_linux: integer[]}, route_sets: object mapping those four names
to integer[8], caveat: string}. Sources: findings/backend-pairs-v1/bf16_ties.json first L0 example;
findings/topk-portability/{macos,linux}.json; corresponding bf16 raw probabilities/routes.
Exact ties are descriptive platform/build behavior, not numerical faults.

## census.json

`phase` string; `results` object keyed r0b/bs8/bs32 copies findings/census-olmoe-prefill/summary.json results:
flips, units, r0_flips, sequences_with_flip, sequences (integer counts); flip_rate, excess_over_r0 (fractions);
ci, excess_ci, sequence_flip_fraction_ci (two-number 95% prompt-cluster bootstrap fraction intervals).
`structure` keyed bs8/bs32 contains tokens, token_layers, differing_sets, one_expert_swaps,
tokens_flipped_any_layer, tokens_flipped_all_layers (counts), per_layer_flips (16 counts), per_layer_rates
(16 fractions), one_expert_share (share among flipped token-layers), token_flip_fraction (share among tokens),
replaced_experts_histogram and flipped_layers_per_token_histogram (count arrays).
Source: findings/census-olmoe-prefill/structure.json results. No bootstrap or rates are rerun.

## accumulator.json

`rows: Row[]`, from findings/accumulator-width/per_op_bits.csv. Row: `accumulator` string
(router/o_proj/rms/expert/combine; source oproj renamed o_proj), `format` string, `mantissa_bits` integer,
`flips_layer0`, `flips_layer7`, `flips_total`, `tokens_per_layer` integer counts;
`calibrated_flagged_L0`, `calibrated_flagged_L7`, `patch_checked` booleans;
`calibrated_localized_L0`, `calibrated_localized_L7`, `strict_first_op_L0`, `strict_first_op_L7` strings;
`router_logit_rel_l2_max`, `output_rel_l2_max` numbers (dimensionless).
Rows include widths 7..23 and source Kahan formats (mantissa_bits=7, format=kahan_e8m7). Total is the sum of the two saved layer counts.

## nccl.json

`ranks: integer[4]`; `tree_order: integer[4]` from order_summary Tree best_order;
`ring_blocks` copies position-pattern.json checks (n, blocks, elements, elements_per_block counts;
algo/target/wire/actual_sha256 strings; ranks integer[], orders integer[][], bit_exact boolean).
`order_summary` copies summary.json order_summary (algo/target/wire/best_tree strings, ranks/best_order integer[],
best_share/best_bin_min/best_bin_max/any_order_share fractions).
`n2` copies summary.json errors filtered n=2 (counts, booleans, hashes, numerical error metrics and definition).
`caveat` string from position-pattern label. Sources: findings/nccl-validation/order-followup/{summary,position-pattern}.json.

## headline.json

`items: {id: string, label: string, values: object, source: string, source_fields: string[]}[]`, five README bullets.
Values retain source names/types/units; census flip_rate is a fraction; evaluation counts are integers;
accumulator minimum and detection widths are mantissa bits; NCCL shares fractions; Qwen route counts integers.
Sources: findings/census-olmoe-prefill/summary.json; findings/routediff-eval-real-v2/summary.json;
findings/accumulator-width/summary.json; findings/nccl-validation/order-followup/summary.json;
findings/qwen3-port/h_port_fp32.json. Labels are editorial, not a source of numerical measurements.

## Machine-checkable shape schema

The following record grammar is normative for exporter-owned fields. Records have exactly the named keys;
`*` maps arbitrary keys to one shape, one-element arrays describe arrays, `|` means union, and names refer to
definitions. Primitive `object` is reserved for unmodified source-owned reports/records, whose values are tested
against their source files. Nullability is explicit. Additional array sizes and domain constraints are tested.

```json
{
  "definitions": {
    "Provenance": {"sources": [{"path": "string", "sha256": "string"}]},
    "Op": {"id": "string", "display_name": "string", "explanation": "string", "shape": ["integer"], "dtype": "string", "pytorch": "string", "cpp": "string"},
    "TokenOp": {"max_scaled_ulp": "number|null", "tolerance_scaled_ulp": "number|null", "compared": "boolean", "flagged": "boolean"},
    "Routes": {"reference": ["integer"], "candidate": ["integer"]},
    "State": {"batch": "object", "source_runs": {"reference": "string", "candidate": "string"}, "token": {"ops": {"*": "TokenOp"}, "route_sets_equal": "boolean"}, "routes": "Routes"},
    "UnavailableState": {"batch": "object", "source_runs": "null", "token": "null", "routes": "null", "unavailable": "string"},
    "Scenario": {"id": "string", "label": "string", "bug_type": "string|null", "source_report": "string", "batch_scope": "string", "token_index": "integer", "before": "State", "after": "State|UnavailableState|null"},
    "AccumulatorRow": {"accumulator": "string", "format": "string", "mantissa_bits": "integer", "flips_layer0": "integer", "flips_layer7": "integer", "flips_total": "integer", "tokens_per_layer": "integer", "router_logit_rel_l2_max": "number", "output_rel_l2_max": "number", "calibrated_flagged_L0": "boolean", "calibrated_flagged_L7": "boolean", "calibrated_localized_L0": "string", "calibrated_localized_L7": "string", "patch_checked": "boolean", "strict_first_op_L0": "string", "strict_first_op_L7": "string"}
  },
  "files": {
    "pipeline.json": {"schema_version": "integer", "provenance": "Provenance", "ops": ["Op"], "edges": [["string"]]},
    "scenarios.json": {"schema_version": "integer", "provenance": "Provenance", "token": {"layer": "integer", "index": "integer", "prompt_id": "string", "position": "integer", "split": "string", "normalized_margin": "number", "probabilities": ["number"], "top8": ["integer"]}, "scenarios": ["Scenario"], "exact_tie": {"layer": "integer", "index": "integer", "probabilities": ["number"], "tied_experts": ["integer"], "choices": {"cpp": ["integer"], "apple_gpu": ["integer"], "pytorch_macos": ["integer"], "pytorch_linux": ["integer"]}, "route_sets": {"cpp": ["integer"], "apple_gpu": ["integer"], "pytorch_macos": ["integer"], "pytorch_linux": ["integer"]}, "caveat": "string"}},
    "census.json": {"schema_version": "integer", "provenance": "Provenance", "phase": "string", "results": "object", "structure": "object"},
    "accumulator.json": {"schema_version": "integer", "provenance": "Provenance", "rows": ["AccumulatorRow"]},
    "nccl.json": {"schema_version": "integer", "provenance": "Provenance", "ranks": ["integer"], "tree_order": ["integer"], "ring_blocks": ["object"], "order_summary": ["object"], "n2": ["object"], "caveat": "string"},
    "headline.json": {"schema_version": "integer", "provenance": "Provenance", "items": [{"id": "string", "label": "string", "values": "object", "source": "string", "source_fields": ["string"]}]}
  }
}
```
