// Coin-Flip Experts: one OLMoE-style MoE block (with the router-input path) with every numerics choice explicit.
//
// Per token:  router_input = RMSNorm(residual + o_proj_in @ o_proj^T)
//             logits -> softmax -> top-k -> SwiGLU experts -> weighted combine
// The defaults reproduce transformers 5.12.1 OlmoeSparseMoeBlock with eager experts in bf16
// (models/olmoe/modeling_olmoe.py:301-375, RMSNorm :58-63). Every op is recorded under the names in
// spec/INTERFACES.md so two runs can be compared op by op.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "numerics.hpp"

namespace coinflip {

struct BlockConfig {
  int hidden = 2048;
  int inter = 1024;
  int n_experts = 64;
  int top_k = 8;
  bool norm_topk_prob = false;
  double rms_eps = 1e-5;
};

// Weights as float32 values (exactly representable in the checkpoint dtype). Row-major.
struct BlockWeights {
  BlockConfig cfg;
  std::vector<float> o_proj;     // [H, H]
  std::vector<float> norm;       // [H]
  std::vector<float> router;     // [E, H]
  std::vector<float> gate_up;    // [E, 2I, H]: rows 0..I-1 gate, I..2I-1 up
  std::vector<float> down;       // [E, H, I]
  static BlockWeights random(const BlockConfig& cfg, uint64_t seed);  // for tests
};

enum class CombineOrder { ExpertAsc, ExpertDesc, Slot };
const char* to_string(CombineOrder o);
CombineOrder parse_combine_order(const std::string& s);  // "expert_asc" "expert_desc" "slot"

enum class CollectiveOn { None, OProj, Down, Both };
CollectiveOn parse_collective_on(const std::string& s);  // "none" "oproj" "down" "both"

struct Numerics {
  DType storage = DType::BF16;         // model dtype: activations, matmul outputs, elementwise results
  AccSpec oproj{DType::FP32, Order::Sequential, 32, true};
  DType rms = DType::FP32;             // RMSNorm internal precision (HF upcasts to fp32)
  DType rms_acc = DType::FP32;         // accumulator of the variance sum (moe_run: defaults to `rms`)
  bool rms_kahan = false;              // variance sum as a sequential Kahan sum in rms_acc
  Order rms_order = Order::TorchCascade;  // order of the variance sum (torch CPU: 4-lane cascade, see numerics.hpp)
  int rms_block = 4;
  AccSpec router{DType::FP32, Order::Sequential, 32, true};
  DType gate_out = DType::BF16;        // dtype of the stored router logits (F.linear output)
  DType softmax = DType::FP32;         // HF: softmax(..., dtype=torch.float)
  bool topk_on_logits = false;         // HF: topk on fp32 probabilities
  bool tiebreak_lower_index = true;    // exact ties go to the lower expert id
  DType weights = DType::BF16;         // routing weights cast before combine (HF: .to(router_logits.dtype))
  AccSpec expert{DType::FP32, Order::Sequential, 32, true};
  DType combine_acc = DType::BF16;     // HF: index_add_ into a model-dtype buffer
  bool combine_kahan = false;          // combine as a Kahan sum in combine_acc (same order)
  CombineOrder combine_order = CombineOrder::ExpertAsc;  // HF: loop over hit experts in ascending id
  Collective collective{};             // emulated tensor-parallel all-reduce
  CollectiveOn collective_on = CollectiveOn::None;

  // Every element of the fp64 oracle: all formats fp64, sequential order, no FMA distinction.
  static Numerics oracle();
};

// A planted bug (written by evaluation/, never read by routediff). `type` must be in bug_registry().
struct Bug {
  std::string id;
  std::string type;
  std::map<std::string, std::string> params;
};
struct BugInfo {
  std::string type;
  std::string injected_op;  // first op (spec/INTERFACES.md order) whose output the bug changes
  std::string description;
};
const std::vector<BugInfo>& bug_registry();

// Per-op output for a block of tokens. Floating ops are [T, ...] flattened; topk_ids is int.
struct Trace {
  int tokens = 0;
  std::map<std::string, std::vector<double>> f;    // op -> values
  std::map<std::string, std::vector<int>> ids;     // "topk_ids" -> [T, K]
  std::map<std::string, std::vector<int>> shape;   // op -> per-token shape (without T)
  std::map<std::string, std::string> stage;        // op -> "selection" | "post-selection"
  std::map<std::string, DType> logical;            // op -> logical dtype
  int patch_skipped_tokens = 0;                    // post-selection patches skipped (top-k sets differ)
};
const std::vector<std::string>& op_order();  // the 13 ops of spec/INTERFACES.md, in order

// Values that replace an op's output (then everything downstream is recomputed). For slot-aligned post-selection
// ops, `ref_ids` gives the expert id of each slot in the patch source; the patch is applied per expert id, only on
// tokens whose top-k set equals the source's.
struct Patch {
  std::string op;
  std::vector<double> values;  // [T, ...] matching op_order shapes
  std::vector<int> ref_ids;    // [T, K], required for slot-aligned ops and topk_weights
};

// Wall time per stage of run_block, summed over tokens (and threads). Index = position in profile_stages().
using Profile = std::vector<double>;
const std::vector<std::string>& profile_stages();

struct RunOptions {
  Numerics num;
  const Bug* bug = nullptr;
  std::vector<Patch> patches;
  int threads = 0;  // 0 = hardware concurrency
  Profile* profile = nullptr;  // if set, filled with seconds per stage
};

// Run the block on tokens [0, T) with inputs o_proj_in [T, H] and residual [T, H].
Trace run_block(const BlockWeights& w, const std::vector<float>& o_proj_in, const std::vector<float>& residual,
                int tokens, const RunOptions& opt);

int topk_set_mismatch(const int* a, const int* b, int k);

}  // namespace coinflip
