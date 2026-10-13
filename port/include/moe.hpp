// Coin-Flip Experts: C++ reference port of one MoE block with configurable numerics.
//
// Every floating-point knob that a real accelerator has to choose is explicit here:
//   - storage dtype of weights/activations   (Numerics::storage)
//   - accumulator dtype of each dot product  (Numerics::accum)
//   - reduction order of each dot product    (Numerics::order, Numerics::block)
//   - router (gate) accumulator dtype        (Numerics::router_accum)
//   - combine accumulator dtype and order    (Numerics::combine_accum, Numerics::combine_order)
// All arithmetic is carried out in double and *rounded* to the chosen format after every
// operation, so a config emulates that format bit-faithfully (round-to-nearest-even).
// The fp64 oracle is simply Numerics::oracle().
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace coinflip {

enum class DType { FP64, FP32, BF16, FP16 };
enum class Order { Sequential, Pairwise, Blocked };

const char* to_string(DType d);
const char* to_string(Order o);

// Round a double to the nearest representable value of `d` (RNE, with overflow -> inf and
// gradual underflow for fp16/bf16/fp32).
double round_to(DType d, double v);

struct Numerics {
  DType storage = DType::FP32;        // weights + activations are rounded to this on load
  DType accum = DType::FP32;          // expert matmul accumulators
  Order order = Order::Sequential;    // expert matmul reduction order
  int block = 32;                     // block size for Order::Blocked
  DType router_accum = DType::FP32;   // gate matmul accumulator
  Order router_order = Order::Sequential;
  DType combine_accum = DType::FP32;  // weighted sum of expert outputs
  Order combine_order = Order::Sequential;  // Sequential = slot order; Pairwise = tree

  static Numerics oracle();
  std::string describe() const;
};

// Dot product of a and b (length n) with explicit accumulator dtype and reduction order.
// Each product is rounded to `acc` before being summed, matching an FMA-free accumulator.
double dot(const double* a, const double* b, int n, DType acc, Order order, int block);

// Sum values with explicit accumulator dtype and order.
double reduce_sum(const std::vector<double>& v, DType acc, Order order, int block);

struct MoEConfig {
  int hidden = 64;        // d_model
  int ffn = 128;          // expert intermediate size
  int n_experts = 16;
  int top_k = 4;
  bool norm_topk_prob = false;  // OLMoE: false, Qwen3-MoE: true
};

struct ExpertWeights {
  // Row-major: gate_proj/up_proj are [ffn][hidden], down_proj is [hidden][ffn].
  std::vector<double> gate_proj, up_proj, down_proj;
};

struct MoEWeights {
  MoEConfig cfg;
  std::vector<double> router;  // [n_experts][hidden]
  std::vector<ExpertWeights> experts;
  static MoEWeights random(const MoEConfig& cfg, uint64_t seed);
};

struct RouteResult {
  std::vector<int> topk_ids;        // sorted by descending score, ties -> lower index
  std::vector<double> topk_weights; // routing weights actually used in combine
  std::vector<double> logits;       // router logits (after router_accum rounding)
  double margin = 0.0;              // logit[k-th] - logit[(k+1)-th]
};

struct MoEOutput {
  RouteResult route;
  std::vector<double> y;  // [hidden]
};

// Route one token (x has `hidden` entries).
RouteResult route(const MoEWeights& w, const double* x, const Numerics& num);

// Full MoE block forward for one token: route -> experts (SwiGLU) -> weighted combine.
MoEOutput forward(const MoEWeights& w, const double* x, const Numerics& num);

// Number of positions where two top-k id sets differ (set difference size).
int topk_set_mismatch(const std::vector<int>& a, const std::vector<int>& b);

}  // namespace coinflip
