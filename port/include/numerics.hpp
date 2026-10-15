// Coin-Flip Experts: floating-point formats, rounding, and reductions with explicit order.
//
// Values are carried in double and rounded to the target format after every operation, so a setting emulates that
// format bit-faithfully (round-to-nearest-even). Rounding is done on the bit pattern and does not depend on the
// current floating-point rounding mode. fp32 accumulation runs on native float hardware (with or without fused
// multiply-add), which is both exact and fast.
#pragma once

#include <string>
#include <vector>

namespace coinflip {

enum class DType { FP64, FP32, BF16, FP16 };

// Reduction order of a sum or dot product.
//   Sequential: ((x0 + x1) + x2) + ...
//   Pairwise:   recursive halves (tree)
//   Blocked:    sequential within blocks of `block`, then sequential over block partials
//   Strided:    `block` lanes; lane j sums elements j, j+block, ... sequentially, then the lanes are summed
//               pairwise. This is the order a SIMD loop with `block` accumulator lanes produces.
enum class Order { Sequential, Pairwise, Blocked, Strided };

const char* to_string(DType d);
const char* to_string(Order o);
DType parse_dtype(const std::string& s);  // "fp64" "fp32" "bf16" "fp16"
Order parse_order(const std::string& s);  // "seq" "pairwise" "blocked" "strided"

// Round to the nearest representable value of `d` (RNE; overflow -> inf; gradual underflow).
double round_to(DType d, double v);
// Reference implementation of round_to via frexp/ldexp, kept for cross-checking the fast path in tests.
double round_to_slow(DType d, double v);
// Size of one unit in the last place of `d` at the magnitude of v (subnormal spacing below the normal range).
double ulp_of(DType d, double v);

// How a dot product is accumulated.
struct AccSpec {
  DType acc = DType::FP32;
  Order order = Order::Sequential;
  int block = 32;
  bool fma = true;  // fused multiply-add per step (Sequential, Blocked, Strided); Pairwise always rounds products
};

double reduce_sum(const double* v, int n, DType acc, Order order, int block);
inline double reduce_sum(const std::vector<double>& v, DType acc, Order order, int block) {
  return reduce_sum(v.data(), static_cast<int>(v.size()), acc, order, block);
}

// Dot product of a float weight row with an activation vector over [lo, hi). Activations must be representable in
// the accumulator format when acc == FP32 (they are: activations are stored in the model dtype).
double dot(const float* w, const double* x, int lo, int hi, const AccSpec& s);

// Emulated cross-chip all-reduce of a row-parallel matvec: the K dimension is split into n contiguous shards, each
// chip computes its partial dot product with `s`, rounds it to the wire dtype, and the n partials are summed in the
// wire dtype. Ring: sequential over chips, starting at chip (shard_of_output + 1) mod n as in a ring reduce-scatter.
// Tree: pairwise over chips 0..n-1.
enum class Topology { Ring, Tree };
const char* to_string(Topology t);
Topology parse_topology(const std::string& s);  // "ring" "tree"

struct Collective {
  int n = 1;
  Topology topo = Topology::Ring;
  DType wire = DType::FP32;
};

// y[r] = round(out, W[r, :] . x) for an [rows, cols] row-major W, optionally through an emulated all-reduce.
void matvec(const float* W, int rows, int cols, const double* x, double* y, const AccSpec& s, DType out,
            const Collective& c = Collective{});

}  // namespace coinflip
