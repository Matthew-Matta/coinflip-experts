# Numbers on this site and where they come from

Every number the site displays comes from `site/data/*.json` (built by Codex from the `findings/` folders; schema in
`DATA_CONTRACT.md`, provenance hashes inside each file) or from the linked `findings/`, `README.md` and `RESULTS.md`. Results are frozen. Display rounding (`fmt` / `pct` in `js/util.js`) preserves the frozen values; the figures also select saved rows and sum the two saved layer counts for their readouts.

## Figures (drawn at runtime from data)

| Shown | Source |
|---|---|
| Figure 1: 64 router probabilities of layer 0 token 651, top 8, #8 = expert 20, #9 = expert 23 | `scenarios.json` `token` |
| Figure 2: per-layer share of token-layers with a different set, batch 8 vs 1 | `census.json` `structure.bs8.per_layer_rates` |
| Figure 3: per-op scaled ULP / tolerance for one token (0 / 502, 0 / 502, 0 / 366, 1,222,418 / 122, 946,387 / 128), first op over tolerance, expert sets (54 instead of 18) | `scenarios.json` `planted_bug.before.token.ops`, `.before.batch.first_flagged_op`, `.before.routes`, `.before.token.route_sets_equal` |
| Figure 3: after patching, 0 of 14 tokens with different experts | `scenarios.json` `planted_bug.after.batch.patch_check.route_mismatch_after`, `.before.batch.tokens` |
| "Try it" tie box: layer 0, token 108, experts 33 and 35, p = 0.02100628 displayed (full precision in tooltip; PyTorch CPU), the expert each of the 4 platforms kept, each platform's 8 experts, caveat | `scenarios.json` `exact_tie` (`probabilities`, `tied_experts`, `choices`, `route_sets`, `caveat`) |
| Figure 4 slider readout: per-width flips (total, layer 0, layer 7), 2,696 tokens, flagged on both / one / no layers | `accumulator.json` `rows[]` (router: `flips_*`, `tokens_per_layer` x 2, `calibrated_flagged_L0/L7`) |
| Figure 4: flips per mantissa width (7-23) for the router accumulator, layers 0 and 7; shaded widths flagged on both layers; "no flips from 18 bits" | `accumulator.json` `rows[]` (router, e8m*) |

## Text (static in `index.html`)

| Shown | Source |
|---|---|
| 64 experts, top 8 | `pipeline.json` (`router_logits` shape [64], `topk_ids` shape [8]); README |
| Figure 1: layer 0, token 651, experts 20 (#8) and 23 (#9), 0.0242630 vs 0.0242627, gap ~1.4e-5 | `scenarios.json` `token.probabilities`, `token.normalized_margin` (rounded) |
| 200 prompts, batch 1 / 8 / 32, 16 layers, one L4 | `census.json`; README |
| 7.25% (batch 8 vs 1), 0% for the repeat run | `census.json` `results.bs8.flip_rate`, `results.r0b.flip_rate`; README |
| 95% CI 7.03-7.48% | `census.json` `results.bs8.ci` |
| 7.21% (batch 32) | `census.json` `results.bs32.flip_rate` |
| 97.9% swap exactly one expert | `census.json` `structure.bs8.one_expert_share` |
| Figure 2 and its caption: 2.6% (layer 0) to 10.6% | `census.json` `structure.bs8.per_layer_rates`; README caption |
| 44/44 effective bugs detected and localised, 0 false alarms in 32 | `headline.json` `debugger`; README |
| Observed no-flip thresholds: 18 router mantissa bits and 19 o_proj bits; bf16 accumulator's 7 bits flip 13%, flags from 21 bits down | `accumulator.json` rows (router: last flips at 17 bits, 360 of 2,696 at 7 bits; o_proj: last flips at 18; both layers flagged up to 21); README |
| NCCL: 2 ranks bit-exact; fitted Tree order and 8 rotating Ring blocks reproduce all 1.2M captured elements; not a unique instruction trace | `nccl.json` `n2`, `tree_order`, `ring_blocks` (131,072 + 1,048,576 elements per wire); README |
| Qwen3-30B-A3B layer (named, no count) | README |
| Qwen3-30B-A3B layer ported in ~15 minutes; 1,361 of 1,361 tokens pick the same experts (fp32) | README ("~15 minutes", "1361/1361"); `headline.json` `qwen` |
| routediff on 4 backends (C++ port, PyTorch CPU, Apple GPU, NVIDIA L4) | README |
| NEON SIMD 2.6-2.9x faster than same-order scalar C++; changes no routes | RESULTS.md (SIMD row) and `findings/simd/` (not in site/data or README) |
| Layers 0 and 7 | README ("OLMoE-1B-7B, 2 layers"); `accumulator.json` (`flips_layer0`, `flips_layer7`) |

Identifiers that are not measurements: SGLang 0.5.21 and upstream PR #40980 (RESULTS.md), the arXiv ids
2607.28097, 2608.11212, 2510.11370, 2609.08368 and the Fireworks post date 2026-09-30 (RESULTS.md related work),
transformers 5.12.1 line numbers (`pipeline.json`).

## Scope qualifications

- The failed port fidelity criterion is 8 ULP at the 99th percentile of element errors, not bit-exactness: `PREREG.md` section 1 and `findings/m2-postfix-validation/`.
- Evaluation footnote: 11/44 route-restoration criteria, 33/44 output-restoration criteria: `findings/routediff-eval-real-v2/README.md`, Limits on interpretation. Near-floor downstream localizations: `findings/accumulator-width/README.md`, Sensitivity floor.
- Qwen adaptation changes (rectangular projection and top-k renormalization summation order): `findings/qwen3-port/README.md`. The approximate 15-minute figure is a reported adaptation time, not an independently repeated benchmark.
- Observed precision thresholds apply to the tested inputs and one narrowed accumulator at a time. BF16 accumulator experiments do not represent every implementation using bf16 operands.
- The SGLang census captured expert IDs, not router scores; neither operation-level attribution nor downstream task quality was measured. Single-expert swaps are consistent with near-ties, not proof of their cause.
- Separate real NCCL checks use 2 and 4 ranks; the larger chip-count sweep and communication cost model remain emulation/prediction.
