#include "numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define COINFLIP_NEON 1
#endif

namespace coinflip {

namespace {
constexpr int kE8Base = 100;  // DType values kE8Base + mant encode e8m<mant>
}

DType e8_format(int mant) {
  if (mant < 1 || mant > 51) throw std::invalid_argument("e8 format needs 1..51 mantissa bits");
  return static_cast<DType>(kE8Base + mant);
}

bool is_e8_format(DType d) { return static_cast<int>(d) > kE8Base && static_cast<int>(d) <= kE8Base + 51; }

int mantissa_bits(DType d) {
  if (is_e8_format(d)) return static_cast<int>(d) - kE8Base;
  switch (d) {
    case DType::FP64: return 52;
    case DType::FP32: return 23;
    case DType::BF16: return 7;
    case DType::FP16: return 10;
  }
  throw std::logic_error("bad dtype");
}

const char* to_string(DType d) {
  if (is_e8_format(d)) {
    static const std::vector<std::string> names = [] {
      std::vector<std::string> v(52);
      for (int m = 1; m <= 51; ++m) v[m] = "e8m" + std::to_string(m);
      return v;
    }();
    return names[mantissa_bits(d)].c_str();
  }
  switch (d) {
    case DType::FP64: return "fp64";
    case DType::FP32: return "fp32";
    case DType::BF16: return "bf16";
    case DType::FP16: return "fp16";
  }
  return "?";
}

const char* to_string(Order o) {
  switch (o) {
    case Order::Sequential: return "seq";
    case Order::Pairwise: return "pairwise";
    case Order::Blocked: return "blocked";
    case Order::Strided: return "strided";
    case Order::TorchCascade: return "torch_cascade";
  }
  return "?";
}

const char* to_string(Topology t) { return t == Topology::Ring ? "ring" : "tree"; }

DType parse_dtype(const std::string& s) {
  if (s == "fp64") return DType::FP64;
  if (s == "fp32") return DType::FP32;
  if (s == "bf16") return DType::BF16;
  if (s == "fp16") return DType::FP16;
  if (s.size() > 3 && s.compare(0, 3, "e8m") == 0 && s.find_first_not_of("0123456789", 3) == std::string::npos)
    return e8_format(std::stoi(s.substr(3)));
  throw std::invalid_argument("unknown dtype: " + s);
}

Order parse_order(const std::string& s) {
  if (s == "seq") return Order::Sequential;
  if (s == "pairwise") return Order::Pairwise;
  if (s == "blocked") return Order::Blocked;
  if (s == "strided") return Order::Strided;
  if (s == "torch_cascade") return Order::TorchCascade;
  throw std::invalid_argument("unknown order: " + s);
}

Topology parse_topology(const std::string& s) {
  if (s == "ring") return Topology::Ring;
  if (s == "tree") return Topology::Tree;
  throw std::invalid_argument("unknown topology: " + s);
}

namespace {

struct Format {
  int mant;  // explicit mantissa bits
  int emin;  // exponent of the smallest normal
  int emax;  // exponent of the largest finite
};

Format format_of(DType d) {
  if (is_e8_format(d)) return {mantissa_bits(d), -126, 127};
  switch (d) {
    case DType::BF16: return {7, -126, 127};
    case DType::FP16: return {10, -14, 15};
    case DType::FP32: return {23, -126, 127};
    case DType::FP64: return {52, -1022, 1023};
  }
  throw std::logic_error("bad dtype");
}

double max_finite(const Format& f) { return std::ldexp(2.0 - std::ldexp(1.0, -f.mant), f.emax); }

// Round-to-nearest-even with integer arithmetic on the quotient, independent of the FP rounding mode.
double rne_integer(double q) {
  const double fl = std::floor(q);
  const double frac = q - fl;  // exact: q has at most 53 significant bits and fl < q < fl + 1
  if (frac > 0.5) return fl + 1.0;
  if (frac < 0.5) return fl;
  return std::fmod(fl, 2.0) == 0.0 ? fl : fl + 1.0;
}

// Fast path for normal-range values of a format with `mant` < 52: RNE on the double's bit pattern.
inline double round_bits(double v, int mant) {
  uint64_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  const int shift = 52 - mant;
  const uint64_t lsb = (bits >> shift) & 1u;
  bits += ((uint64_t{1} << (shift - 1)) - 1) + lsb;  // carry into the exponent is the correct result
  bits &= ~((uint64_t{1} << shift) - 1);
  double out;
  std::memcpy(&out, &bits, sizeof out);
  return out;
}

}  // namespace

double round_to_slow(DType d, double v) {
  if (d == DType::FP64 || std::isnan(v) || std::isinf(v) || v == 0.0) return v;
  const Format f = format_of(d);
  const double a = std::fabs(v);
  int e;
  std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
  int exp = e - 1;
  if (exp < f.emin) exp = f.emin;  // subnormal: fixed quantum
  const double quantum = std::ldexp(1.0, exp - f.mant);
  double out = rne_integer(a / quantum) * quantum;
  if (out > max_finite(f)) out = std::numeric_limits<double>::infinity();
  return std::copysign(out, v);
}

double round_to(DType d, double v) {
  if (is_e8_format(d)) {
    const Format f = format_of(d);
    const double a = std::fabs(v);
    if (a >= std::ldexp(1.0, f.emin) && a < std::ldexp(1.0, f.emax)) return round_bits(v, f.mant);
    return round_to_slow(d, v);
  }
  switch (d) {
    case DType::FP64: return v;
    case DType::FP32: return static_cast<double>(static_cast<float>(v));  // IEEE conversion is RNE
    case DType::BF16:
    case DType::FP16: {
      const Format f = format_of(d);
      const double a = std::fabs(v);
      // Fast path only strictly inside the normal range; edges (subnormal, overflow, inf/nan) go the slow way.
      if (a >= std::ldexp(1.0, f.emin) && a < std::ldexp(1.0, f.emax)) return round_bits(v, f.mant);
      return round_to_slow(d, v);
    }
  }
  throw std::logic_error("bad dtype");
}

double ulp_of(DType d, double v) {
  const Format f = format_of(d);
  const double a = std::fabs(v);
  int e;
  std::frexp(a == 0.0 ? std::ldexp(1.0, f.emin) : a, &e);
  int exp = std::max(e - 1, f.emin);
  return std::ldexp(1.0, exp - f.mant);
}

namespace {

int ceil_log2(long long n) {
  int r = 0;
  while ((1LL << r) < n) ++r;
  return r;
}

// PyTorch CPU sum of a contiguous row (SumKernel.cpp): V-lane vectors, ILP 4 over vectors, 4-level cascade.
double torch_cascade(const double* v, int n, DType acc, int V) {
  constexpr int ilp = 4, levels = 4;
  if (V <= 0) throw std::invalid_argument("torch_cascade needs block (lanes) > 0");
  const int vec_size = n / V, size_ilp = vec_size / ilp;
  const int level_power = std::max(4, ceil_log2(size_ilp) / levels);
  const long long step = 1LL << level_power, mask0 = step - 1;
  std::vector<double> a(static_cast<size_t>(levels) * ilp * V, 0.0);  // [level][ilp][lane]
  auto A = [&](int l, int k, int lane) -> double& { return a[(static_cast<size_t>(l) * ilp + k) * V + lane]; };
  auto add_row = [&](long long i) {
    for (int k = 0; k < ilp; ++k)
      for (int lane = 0; lane < V; ++lane) A(0, k, lane) = round_to(acc, A(0, k, lane) + v[(i * ilp + k) * V + lane]);
  };
  long long i = 0;
  while (i + step <= size_ilp) {
    for (long long j = 0; j < step; ++j, ++i) add_row(i);
    for (int j = 1; j < levels; ++j) {
      for (int k = 0; k < ilp; ++k)
        for (int lane = 0; lane < V; ++lane) {
          A(j, k, lane) = round_to(acc, A(j, k, lane) + A(j - 1, k, lane));
          A(j - 1, k, lane) = 0.0;
        }
      if ((i & (mask0 << (j * level_power))) != 0) break;
    }
  }
  for (; i < size_ilp; ++i) add_row(i);
  for (int j = 1; j < levels; ++j)
    for (int k = 0; k < ilp; ++k)
      for (int lane = 0; lane < V; ++lane) A(0, k, lane) = round_to(acc, A(0, k, lane) + A(j, k, lane));
  for (int vi = size_ilp * ilp; vi < vec_size; ++vi)  // leftover whole vectors go to partial 0
    for (int lane = 0; lane < V; ++lane) A(0, 0, lane) = round_to(acc, A(0, 0, lane) + v[vi * V + lane]);
  std::vector<double> s(V);
  for (int lane = 0; lane < V; ++lane) {
    s[lane] = A(0, 0, lane);
    for (int k = 1; k < ilp; ++k) s[lane] = round_to(acc, s[lane] + A(0, k, lane));
  }
  double out = 0.0;
  for (int e = vec_size * V; e < n; ++e) out = round_to(acc, out + v[e]);  // scalar tail first, as in torch
  for (int lane = 0; lane < V; ++lane) out = round_to(acc, out + s[lane]);
  return out;
}

double sum_range(const double* v, int lo, int hi, DType acc, Order order, int block) {
  const int n = hi - lo;
  if (n <= 0) return 0.0;
  switch (order) {
    case Order::Sequential: {
      double s = 0.0;
      for (int i = lo; i < hi; ++i) s = round_to(acc, s + v[i]);
      return s;
    }
    case Order::Pairwise: {
      if (n == 1) return round_to(acc, v[lo]);
      const int mid = lo + n / 2;
      return round_to(acc, sum_range(v, lo, mid, acc, order, block) + sum_range(v, mid, hi, acc, order, block));
    }
    case Order::Blocked: {
      if (block <= 0) throw std::invalid_argument("block must be > 0");
      double s = 0.0;
      for (int b = lo; b < hi; b += block)
        s = round_to(acc, s + sum_range(v, b, std::min(b + block, hi), acc, Order::Sequential, block));
      return s;
    }
    case Order::Strided: {
      if (block <= 0) throw std::invalid_argument("block (lanes) must be > 0");
      std::vector<double> lanes(std::min(block, n), 0.0);
      for (int i = lo; i < hi; ++i) {
        double& l = lanes[(i - lo) % lanes.size()];
        l = round_to(acc, l + v[i]);
      }
      return sum_range(lanes.data(), 0, static_cast<int>(lanes.size()), acc, Order::Pairwise, block);
    }
    case Order::TorchCascade:
      return torch_cascade(v + lo, n, acc, block);
  }
  throw std::logic_error("bad order");
}

// fp32 accumulation on native floats. With fma the product is not rounded before the add.
float dot_f32(const float* w, const float* x, int lo, int hi, const AccSpec& s) {
  auto step = [&](float acc, int i) { return s.fma ? std::fmaf(w[i], x[i], acc) : acc + w[i] * x[i]; };
  switch (s.order) {
    case Order::Sequential: {
      float acc = 0.0f;
      for (int i = lo; i < hi; ++i) acc = step(acc, i);
      return acc;
    }
    case Order::Blocked: {
      float acc = 0.0f;
      for (int b = lo; b < hi; b += s.block) {
        float part = 0.0f;
        for (int i = b; i < std::min(b + s.block, hi); ++i) part = step(part, i);
        acc = acc + part;
      }
      return acc;
    }
    case Order::Strided: {
      const int L = std::min(s.block, std::max(hi - lo, 1));
      std::vector<double> lanes(L, 0.0);
      std::vector<float> lf(L, 0.0f);
      for (int i = lo; i < hi; ++i) {
        float& l = lf[(i - lo) % L];
        l = step(l, i);
      }
      for (int j = 0; j < L; ++j) lanes[j] = lf[j];
      return static_cast<float>(sum_range(lanes.data(), 0, L, DType::FP32, Order::Pairwise, L));
    }
    case Order::Pairwise:
    case Order::TorchCascade: {
      std::vector<double> p(hi - lo);
      for (int i = lo; i < hi; ++i) p[i - lo] = static_cast<float>(w[i] * x[i]);  // rounded product
      return static_cast<float>(sum_range(p.data(), 0, hi - lo, DType::FP32, s.order, s.block));
    }
  }
  throw std::logic_error("bad order");
}

// Generic accumulation in double with rounding to `acc` after each step (FP64, BF16, FP16). Products of inputs that
// are at most fp32-precise are exact in double.
double dot_generic(const float* w, const double* x, int lo, int hi, const AccSpec& s) {
  const DType acc = s.acc;
  auto prod = [&](int i) {
    const double p = static_cast<double>(w[i]) * x[i];
    return s.fma ? p : round_to(acc, p);
  };
  if (s.kahan) {
    if (s.order != Order::Sequential) throw std::invalid_argument("kahan needs sequential order");
    double a = 0.0, c = 0.0;
    for (int i = lo; i < hi; ++i) {
      const double y = round_to(acc, prod(i) - c);
      const double t = round_to(acc, a + y);
      c = round_to(acc, round_to(acc, t - a) - y);
      a = t;
    }
    return a;
  }
  switch (s.order) {
    case Order::Sequential: {
      double a = 0.0;
      for (int i = lo; i < hi; ++i) a = round_to(acc, a + prod(i));
      return a;
    }
    case Order::Blocked: {
      double a = 0.0;
      for (int b = lo; b < hi; b += s.block) {
        double part = 0.0;
        for (int i = b; i < std::min(b + s.block, hi); ++i) part = round_to(acc, part + prod(i));
        a = round_to(acc, a + part);
      }
      return a;
    }
    case Order::Strided: {
      const int L = std::min(s.block, std::max(hi - lo, 1));
      std::vector<double> lanes(L, 0.0);
      for (int i = lo; i < hi; ++i) {
        double& l = lanes[(i - lo) % L];
        l = round_to(acc, l + prod(i));
      }
      return sum_range(lanes.data(), 0, L, acc, Order::Pairwise, L);
    }
    case Order::Pairwise:
    case Order::TorchCascade: {
      std::vector<double> p(hi - lo);
      for (int i = lo; i < hi; ++i) p[i - lo] = round_to(acc, static_cast<double>(w[i]) * x[i]);
      return sum_range(p.data(), 0, hi - lo, acc, s.order, s.block);
    }
  }
  throw std::logic_error("bad order");
}

}  // namespace

bool simd_is_native() {
#ifdef COINFLIP_NEON
  return true;
#else
  return false;
#endif
}

namespace {

// Portable loop with exactly the NEON order (see AccSpec::simd).
float dot_simd_portable(const float* w, const float* x, int n, int accs) {
  float lane[8][4] = {};
  const int step = 4 * accs;
  int i = 0;
  for (; i + step <= n; i += step)
    for (int a = 0; a < accs; ++a)
      for (int l = 0; l < 4; ++l) lane[a][l] = std::fmaf(w[i + 4 * a + l], x[i + 4 * a + l], lane[a][l]);
  for (; i + 4 <= n; i += 4)
    for (int l = 0; l < 4; ++l) lane[0][l] = std::fmaf(w[i + l], x[i + l], lane[0][l]);
  for (int width = 1; width < accs; width *= 2)  // balanced tree over accumulators
    for (int a = 0; a + width < accs; a += 2 * width)
      for (int l = 0; l < 4; ++l) lane[a][l] = lane[a][l] + lane[a + width][l];
  float s = (lane[0][0] + lane[0][1]) + (lane[0][2] + lane[0][3]);
  for (; i < n; ++i) s = std::fmaf(w[i], x[i], s);
  return s;
}

#ifdef COINFLIP_NEON
float dot_simd_neon(const float* w, const float* x, int n, int accs) {
  float32x4_t v[8];
  for (auto& vv : v) vv = vdupq_n_f32(0.f);
  const int step = 4 * accs;
  int i = 0;
  if (accs == 4) {
    for (; i + 16 <= n; i += 16) {
      v[0] = vfmaq_f32(v[0], vld1q_f32(w + i), vld1q_f32(x + i));
      v[1] = vfmaq_f32(v[1], vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
      v[2] = vfmaq_f32(v[2], vld1q_f32(w + i + 8), vld1q_f32(x + i + 8));
      v[3] = vfmaq_f32(v[3], vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
    }
  } else if (accs == 8) {
    for (; i + 32 <= n; i += 32) {
      v[0] = vfmaq_f32(v[0], vld1q_f32(w + i), vld1q_f32(x + i));
      v[1] = vfmaq_f32(v[1], vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
      v[2] = vfmaq_f32(v[2], vld1q_f32(w + i + 8), vld1q_f32(x + i + 8));
      v[3] = vfmaq_f32(v[3], vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
      v[4] = vfmaq_f32(v[4], vld1q_f32(w + i + 16), vld1q_f32(x + i + 16));
      v[5] = vfmaq_f32(v[5], vld1q_f32(w + i + 20), vld1q_f32(x + i + 20));
      v[6] = vfmaq_f32(v[6], vld1q_f32(w + i + 24), vld1q_f32(x + i + 24));
      v[7] = vfmaq_f32(v[7], vld1q_f32(w + i + 28), vld1q_f32(x + i + 28));
    }
  } else {
    for (; i + step <= n; i += step)
      for (int a = 0; a < accs; ++a) v[a] = vfmaq_f32(v[a], vld1q_f32(w + i + 4 * a), vld1q_f32(x + i + 4 * a));
  }
  for (; i + 4 <= n; i += 4) v[0] = vfmaq_f32(v[0], vld1q_f32(w + i), vld1q_f32(x + i));
  for (int width = 1; width < accs; width *= 2)
    for (int a = 0; a + width < accs; a += 2 * width) v[a] = vaddq_f32(v[a], v[a + width]);
  float s = vaddvq_f32(v[0]);  // FADDP: (l0 + l1) + (l2 + l3)
  for (; i < n; ++i) s = std::fmaf(w[i], x[i], s);
  return s;
}
#endif

}  // namespace

float dot_simd(const float* w, const float* x, int n, int accs, bool force_portable) {
  if (accs != 1 && accs != 2 && accs != 4 && accs != 8) throw std::invalid_argument("simd_accs must be 1, 2, 4 or 8");
#ifdef COINFLIP_NEON
  if (!force_portable) return dot_simd_neon(w, x, n, accs);
#endif
  (void)force_portable;
  return dot_simd_portable(w, x, n, accs);
}

double reduce_sum(const double* v, int n, DType acc, Order order, int block) {
  return sum_range(v, 0, n, acc, order, block);
}

double kahan_sum(const double* v, int n, DType acc) {
  double a = 0.0, c = 0.0;
  for (int i = 0; i < n; ++i) {
    const double y = round_to(acc, v[i] - c);
    const double t = round_to(acc, a + y);
    c = round_to(acc, round_to(acc, t - a) - y);
    a = t;
  }
  return a;
}

double dot(const float* w, const double* x, int lo, int hi, const AccSpec& s) {
  if (s.acc == DType::FP32 && !s.kahan) {
    std::vector<float> xf(x + lo, x + hi);
    return dot_f32(w + lo, xf.data(), 0, hi - lo, s);
  }
  return dot_generic(w, x, lo, hi, s);
}

void matvec(const float* W, int rows, int cols, const double* x, double* y, const AccSpec& s, DType out,
            const Collective& c) {
  if (c.n < 1) throw std::invalid_argument("collective n must be >= 1");
  std::vector<float> xf;
  const bool native = s.acc == DType::FP32 && !s.kahan;
  if (native) xf.assign(x, x + cols);
  if (s.simd && !native) throw std::invalid_argument("simd requires a plain fp32 accumulator");
  auto partial = [&](const float* row, int lo, int hi) -> double {
    if (s.simd) return dot_simd(row + lo, xf.data() + lo, hi - lo, s.simd_accs);
    if (native) return dot_f32(row, xf.data(), lo, hi, s);
    return dot_generic(row, x, lo, hi, s);
  };
  const int shard = (cols + c.n - 1) / c.n;
  std::vector<double> parts(c.n);
  for (int r = 0; r < rows; ++r) {
    const float* row = W + static_cast<size_t>(r) * cols;
    if (c.n == 1) {
      y[r] = round_to(out, partial(row, 0, cols));
      continue;
    }
    for (int k = 0; k < c.n; ++k) {
      const int lo = std::min(k * shard, cols), hi = std::min(lo + shard, cols);
      parts[k] = round_to(c.wire, partial(row, lo, hi));
    }
    double total;
    if (c.topo == Topology::Tree) {
      total = reduce_sum(parts.data(), c.n, c.wire, Order::Pairwise, 1);
    } else {
      // Ring reduce-scatter: output chunk j is accumulated starting at chip (j + 1) mod n.
      const int chunk = static_cast<int>(static_cast<long long>(r) * c.n / rows);
      const int start = (chunk + 1) % c.n;
      double acc = 0.0;
      for (int k = 0; k < c.n; ++k) acc = round_to(c.wire, acc + parts[(start + k) % c.n]);
      total = acc;
    }
    y[r] = round_to(out, total);
  }
}

}  // namespace coinflip
