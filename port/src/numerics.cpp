#include "numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace coinflip {

const char* to_string(DType d) {
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
  }
  return "?";
}

const char* to_string(Topology t) { return t == Topology::Ring ? "ring" : "tree"; }

DType parse_dtype(const std::string& s) {
  if (s == "fp64") return DType::FP64;
  if (s == "fp32") return DType::FP32;
  if (s == "bf16") return DType::BF16;
  if (s == "fp16") return DType::FP16;
  throw std::invalid_argument("unknown dtype: " + s);
}

Order parse_order(const std::string& s) {
  if (s == "seq") return Order::Sequential;
  if (s == "pairwise") return Order::Pairwise;
  if (s == "blocked") return Order::Blocked;
  if (s == "strided") return Order::Strided;
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
    case Order::Pairwise: {
      std::vector<double> p(hi - lo);
      for (int i = lo; i < hi; ++i) p[i - lo] = static_cast<float>(w[i] * x[i]);  // rounded product
      return static_cast<float>(sum_range(p.data(), 0, hi - lo, DType::FP32, Order::Pairwise, s.block));
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
    case Order::Pairwise: {
      std::vector<double> p(hi - lo);
      for (int i = lo; i < hi; ++i) p[i - lo] = round_to(acc, static_cast<double>(w[i]) * x[i]);
      return sum_range(p.data(), 0, hi - lo, acc, Order::Pairwise, s.block);
    }
  }
  throw std::logic_error("bad order");
}

}  // namespace

double reduce_sum(const double* v, int n, DType acc, Order order, int block) {
  return sum_range(v, 0, n, acc, order, block);
}

double dot(const float* w, const double* x, int lo, int hi, const AccSpec& s) {
  if (s.acc == DType::FP32) {
    std::vector<float> xf(x + lo, x + hi);
    return dot_f32(w + lo, xf.data(), 0, hi - lo, s);
  }
  return dot_generic(w, x, lo, hi, s);
}

void matvec(const float* W, int rows, int cols, const double* x, double* y, const AccSpec& s, DType out,
            const Collective& c) {
  if (c.n < 1) throw std::invalid_argument("collective n must be >= 1");
  std::vector<float> xf;
  if (s.acc == DType::FP32) xf.assign(x, x + cols);
  auto partial = [&](const float* row, int lo, int hi) -> double {
    if (s.acc == DType::FP32) return dot_f32(row, xf.data(), lo, hi, s);
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
