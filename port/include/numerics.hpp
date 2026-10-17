// Coin-Flip Experts: floating-point formats, rounding, and reductions with explicit order.
//
// Values are carried in double and rounded to the target format after every operation, so a setting emulates that
// format bit-faithfully (round-to-nearest-even). bf16, fp16 and e8m<m> rounding is done on the bit pattern and does
// not depend on the current floating-point rounding mode. fp32 rounding and fp32 accumulation use native float
// hardware (with or without fused multiply-add), which follows the FP environment (round-to-nearest by default).
#pragma once

#include <string>
#include <vector>

namespace coinflip {

enum class DType : int { FP64, FP32, BF16, FP16 };

// Parametric formats with fp32's 8-bit exponent and `mant` explicit mantissa bits (1..51), written "e8m<mant>" (used
// by the accumulator-width sweep). They always take the emulated path (double, rounded after every operation), so
// e8m23 rounds exactly like fp32 and e8m7 exactly like bf16.
DType e8_format(int mant);
bool is_e8_format(DType d);
int mantissa_bits(DType d);

// Reduction order of a sum or dot product.
//   Sequential: ((x0 + x1) + x2) + ...
//   Pairwise:   recursive halves (tree)
//   Blocked:    sequential within blocks of `block`, then sequential over block partials
//   Strided:    `block` lanes; lane j sums elements j, j+block, ... sequentially, then the lanes are summed
//               pairwise. This is the order a SIMD loop with `block` accumulator lanes produces.
//   TorchCascade: PyTorch's CPU sum over a contiguous row (aten/src/ATen/native/cpu/SumKernel.cpp,
//               vectorized_inner_sum -> row_sum -> multi_row_sum) with `block`-lane vectors: 4-way ILP over
//               vectors, 4-level cascade over chunks, then the ILP partials and finally the lanes summed in order.
//               With 4 lanes it reproduces torch's fp32 RMSNorm variance bit-exactly on OLMoE layer 0 (M2).
enum class Order { Sequential, Pairwise, Blocked, Strided, TorchCascade };

const char* to_string(DType d);
const char* to_string(Order o);
DType parse_dtype(const std::string& s);  // "fp64" "fp32" "bf16" "fp16" "e8m<mant>"
Order parse_order(const std::string& s);  // "seq" "pairwise" "blocked" "strided" "torch_cascade"

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
  // fp32 SIMD path (NEON on AArch64, an order-identical portable loop elsewhere). Overrides `order` and `fma`:
  // `simd_accs` (1, 2, 4 or 8) 4-lane accumulators with FMA; accumulator a lane l takes elements i*4*accs + 4*a + l;
  // leftover whole vectors go to accumulator 0, then accumulators are summed as a balanced tree ((a0+a1)+(a2+a3)), the 4 lanes as
  // (l0+l1)+(l2+l3) (AArch64 FADDP), and a scalar tail is added last with FMA.
  bool simd = false;
  int simd_accs = 4;
  // Kahan (compensated) summation in the accumulator format, sequential order only; always on the emulated path.
  bool kahan = false;
};

// fp32 SIMD dot product in the order documented on AccSpec::simd. `force_portable` runs the portable loop even where
// NEON is available (used by tests to prove the two agree bit for bit).
float dot_simd(const float* w, const float* x, int n, int accs, bool force_portable = false);
bool simd_is_native();

double reduce_sum(const double* v, int n, DType acc, Order order, int block);
// Sequential Kahan sum, every operation rounded to `acc`.
double kahan_sum(const double* v, int n, DType acc);
inline double reduce_sum(const std::vector<double>& v, DType acc, Order order, int block) {
  return reduce_sum(v.data(), static_cast<int>(v.size()), acc, order, block);
}

// Dot product of a float weight row with an activation vector over [lo, hi). Activations must be representable in
// the accumulator format when acc == FP32 (they are: activations are stored in the model dtype). Ignores `s.simd`
// (use dot_simd, or matvec, which honours it).
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
