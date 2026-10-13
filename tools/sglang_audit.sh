#!/usr/bin/env bash
# Lists every place in SGLang's MoE path where numerics can depend on batch size,
# so attribution knows what to patch. Output is a starting point for reading the code,
# not a conclusion: open each hit and record file:line + what decides the choice in NOTES.md.
#
# Usage: tools/sglang_audit.sh [git-ref]   (default: main)
set -euo pipefail
REF="${1:-main}"
DEST="third_party/sglang"
mkdir -p third_party logs
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/sgl-project/sglang.git "$DEST"
fi
SRC="$DEST/python/sglang/srt"
OUT="logs/sglang_audit.txt"
GREP="grep -rnE --include=*.py"
command -v rg >/dev/null && GREP="rg -n --type py -e"

{
  echo "# SGLang audit @ $(git -C "$DEST" rev-parse --short HEAD) ($(date -u +%F))"
  echo
  echo "## 1. Files to read in full"
  for f in layers/moe/topk.py \
           layers/moe/fused_moe_triton/fused_moe.py \
           layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py \
           batch_invariant_ops/batch_invariant_ops.py \
           models/olmoe.py models/qwen3_moe.py \
           distributed/parallel_state.py; do
    if [ -f "$SRC/$f" ]; then echo "  [ok]      $f ($(wc -l < "$SRC/$f") lines)"; else echo "  [MISSING] $f  -> path changed, find it"; fi
  done
  echo "  state capturers:"; ls "$SRC"/state_capturer 2>/dev/null | sed 's/^/    /' || echo "    [MISSING] state_capturer/"
  echo
  echo "## 2. Kernel configs chosen from the number of tokens (batch-size dependent tiling)"
  $GREP "get_config|try_get_optimal_moe_config|BLOCK_SIZE_M|num_tokens|M <=|M >" "$SRC/layers/moe" || true
  echo
  echo "## 3. Split-K / split-KV / reduction choices"
  $GREP "split_k|SPLIT_K|num_kv_splits|split_kv|num_splits" "$SRC" || true
  echo
  echo "## 4. Deterministic / batch-invariant switches"
  $GREP "enable_deterministic_inference|batch_invariant|deterministic" "$SRC" || true
  echo
  echo "## 5. Router / gate precision"
  $GREP "\\.float\\(\\)|router_logits|gate\\(|ReplicatedLinear" "$SRC/layers/moe/topk.py" "$SRC/models/olmoe.py" "$SRC/models/qwen3_moe.py" || true
  echo
  echo "## 6. Combine (moe_sum / reduce) and all-reduce"
  $GREP "moe_sum|sum_reduce|all_reduce|reduce_scatter|custom_all_reduce" "$SRC/layers/moe" "$SRC/distributed" || true
  echo
  echo "## 7. Routed-expert capture (what the existing API returns)"
  $GREP "routed_experts|return_routed_experts" "$SRC" || true
} > "$OUT"

echo "wrote $OUT ($(wc -l < "$OUT") lines)"
