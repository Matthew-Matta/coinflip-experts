#include "moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
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
  }
  return "?";
}

namespace {

// Round v to a binary format with `mant` explicit mantissa bits and exponent range
// [emin, emax] (emin = exponent of the smallest normal). Round-to-nearest-even.
double round_format(double v, int mant, int emin, int emax) {
  if (std::isnan(v) || std::isinf(v) || v == 0.0) return v;
  const double a = std::fabs(v);
  int e;
  std::frexp(a, &e);  // a = f * 2^e, f in [0.5, 1)
  int exp = e - 1;    // a in [2^exp, 2^(exp+1))
  if (exp < emin) exp = emin;  // subnormal: fixed quantum
  const double quantum = std::ldexp(1.0, exp - mant);
  double q = a / quantum;
  double r = std::nearbyint(q);  // default rounding mode is RNE
  double out = r * quantum;
  const double max_finite = std::ldexp(2.0 - std::ldexp(1.0, -mant), emax);
  if (out > max_finite) out = std::numeric_limits<double>::infinity();
  return std::copysign(out, v);
}

}  // namespace

double round_to(DType d, double v) {
  switch (d) {
    case DType::FP64: return v;
    case DType::FP32: return static_cast<double>(static_cast<float>(v));
    case DType::BF16: return round_format(v, 7, -126, 127);
    case DType::FP16: return round_format(v, 10, -14, 15);
  }
  throw std::logic_error("bad dtype");
}

Numerics Numerics::oracle() {
  Numerics n;
  n.storage = n.accum = n.router_accum = n.combine_accum = DType::FP64;
  n.order = n.router_order = n.combine_order = Order::Sequential;
  return n;
}

std::string Numerics::describe() const {
  std::ostringstream s;
  s << "storage=" << to_string(storage) << " accum=" << to_string(accum) << "/" << to_string(order);
  if (order == Order::Blocked) s << block;
  s << " router=" << to_string(router_accum) << "/" << to_string(router_order)
    << " combine=" << to_string(combine_accum) << "/" << to_string(combine_order);
  return s.str();
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
      if (n == 1) return v[lo];
      const int mid = lo + n / 2;
      return round_to(acc, sum_range(v, lo, mid, acc, order, block) +
                               sum_range(v, mid, hi, acc, order, block));
    }
    case Order::Blocked: {
      if (block <= 0) throw std::invalid_argument("block must be > 0");
      double s = 0.0;
      for (int b = lo; b < hi; b += block) {
        const double part = sum_range(v, b, std::min(b + block, hi), acc, Order::Sequential, block);
        s = round_to(acc, s + part);
      }
      return s;
    }
  }
  throw std::logic_error("bad order");
}

}  // namespace

double reduce_sum(const std::vector<double>& v, DType acc, Order order, int block) {
  return sum_range(v.data(), 0, static_cast<int>(v.size()), acc, order, block);
}

double dot(const double* a, const double* b, int n, DType acc, Order order, int block) {
  std::vector<double> prods(n);
  for (int i = 0; i < n; ++i) prods[i] = round_to(acc, a[i] * b[i]);
  return sum_range(prods.data(), 0, n, acc, order, block);
}

MoEWeights MoEWeights::random(const MoEConfig& cfg, uint64_t seed) {
  MoEWeights w;
  w.cfg = cfg;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  const double sh = 1.0 / std::sqrt(static_cast<double>(cfg.hidden));
  const double sf = 1.0 / std::sqrt(static_cast<double>(cfg.ffn));
  w.router.resize(static_cast<size_t>(cfg.n_experts) * cfg.hidden);
  for (auto& x : w.router) x = nd(rng) * sh;
  w.experts.resize(cfg.n_experts);
  for (auto& e : w.experts) {
    e.gate_proj.resize(static_cast<size_t>(cfg.ffn) * cfg.hidden);
    e.up_proj.resize(static_cast<size_t>(cfg.ffn) * cfg.hidden);
    e.down_proj.resize(static_cast<size_t>(cfg.hidden) * cfg.ffn);
    for (auto& x : e.gate_proj) x = nd(rng) * sh;
    for (auto& x : e.up_proj) x = nd(rng) * sh;
    for (auto& x : e.down_proj) x = nd(rng) * sf;
  }
  return w;
}

namespace {

std::vector<double> stored(const double* x, int n, DType d) {
  std::vector<double> out(x, x + n);
  for (auto& v : out) v = round_to(d, v);
  return out;
}

// y[r] = W[r,:] . x for an [rows][cols] row-major matrix, with storage rounding on W.
std::vector<double> matvec(const std::vector<double>& W, const std::vector<double>& x, int rows,
                           int cols, DType storage, DType acc, Order order, int block) {
  std::vector<double> y(rows);
  std::vector<double> row(cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) row[c] = round_to(storage, W[static_cast<size_t>(r) * cols + c]);
    y[r] = round_to(storage, dot(row.data(), x.data(), cols, acc, order, block));
  }
  return y;
}

}  // namespace

RouteResult route(const MoEWeights& w, const double* x_in, const Numerics& num) {
  const auto& cfg = w.cfg;
  if (cfg.top_k >= cfg.n_experts) throw std::invalid_argument("top_k must be < n_experts");
  const auto x = stored(x_in, cfg.hidden, num.storage);
  RouteResult r;
  r.logits.resize(cfg.n_experts);
  std::vector<double> row(cfg.hidden);
  for (int e = 0; e < cfg.n_experts; ++e) {
    for (int c = 0; c < cfg.hidden; ++c)
      row[c] = round_to(num.storage, w.router[static_cast<size_t>(e) * cfg.hidden + c]);
    r.logits[e] = round_to(num.router_accum,
                           dot(row.data(), x.data(), cfg.hidden, num.router_accum, num.router_order, num.block));
  }
  // Softmax in the router accumulator precision (max-subtracted).
  const double mx = *std::max_element(r.logits.begin(), r.logits.end());
  std::vector<double> p(cfg.n_experts);
  double z = 0.0;
  for (int e = 0; e < cfg.n_experts; ++e) {
    p[e] = round_to(num.router_accum, std::exp(r.logits[e] - mx));
    z = round_to(num.router_accum, z + p[e]);
  }
  for (auto& v : p) v = round_to(num.router_accum, v / z);

  // Deterministic top-k: descending logit, ties broken by lower expert index.
  std::vector<int> idx(cfg.n_experts);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return r.logits[a] > r.logits[b]; });
  r.topk_ids.assign(idx.begin(), idx.begin() + cfg.top_k);
  r.margin = r.logits[idx[cfg.top_k - 1]] - r.logits[idx[cfg.top_k]];

  r.topk_weights.resize(cfg.top_k);
  double s = 0.0;
  for (int i = 0; i < cfg.top_k; ++i) {
    r.topk_weights[i] = p[r.topk_ids[i]];
    s = round_to(num.router_accum, s + r.topk_weights[i]);
  }
  if (cfg.norm_topk_prob)
    for (auto& v : r.topk_weights) v = round_to(num.router_accum, v / s);
  return r;
}

MoEOutput forward(const MoEWeights& w, const double* x_in, const Numerics& num) {
  const auto& cfg = w.cfg;
  MoEOutput out;
  out.route = route(w, x_in, num);
  const auto x = stored(x_in, cfg.hidden, num.storage);

  std::vector<std::vector<double>> expert_out(cfg.top_k);
  for (int s = 0; s < cfg.top_k; ++s) {
    const auto& e = w.experts[out.route.topk_ids[s]];
    auto g = matvec(e.gate_proj, x, cfg.ffn, cfg.hidden, num.storage, num.accum, num.order, num.block);
    auto u = matvec(e.up_proj, x, cfg.ffn, cfg.hidden, num.storage, num.accum, num.order, num.block);
    std::vector<double> h(cfg.ffn);
    for (int i = 0; i < cfg.ffn; ++i) {
      const double silu = g[i] / (1.0 + std::exp(-g[i]));
      h[i] = round_to(num.storage, round_to(num.storage, silu) * u[i]);
    }
    expert_out[s] = matvec(e.down_proj, h, cfg.hidden, cfg.ffn, num.storage, num.accum, num.order, num.block);
  }

  out.y.resize(cfg.hidden);
  std::vector<double> terms(cfg.top_k);
  for (int d = 0; d < cfg.hidden; ++d) {
    for (int s = 0; s < cfg.top_k; ++s)
      terms[s] = round_to(num.combine_accum, out.route.topk_weights[s] * expert_out[s][d]);
    out.y[d] = round_to(num.storage, reduce_sum(terms, num.combine_accum, num.combine_order, num.block));
  }
  return out;
}

int topk_set_mismatch(const std::vector<int>& a, const std::vector<int>& b) {
  int miss = 0;
  for (int x : a)
    if (std::find(b.begin(), b.end(), x) == b.end()) ++miss;
  return miss;
}

}  // namespace coinflip
