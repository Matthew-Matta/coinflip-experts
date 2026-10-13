// Minimal self-contained tests (no framework dependency). Exit code != 0 on failure.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "moe.hpp"

using namespace coinflip;

static int failures = 0;
#define CHECK(cond, msg)                                         \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                \
    }                                                            \
  } while (0)

static void test_rounding() {
  CHECK(round_to(DType::FP16, 1.0) == 1.0, "fp16 exact 1");
  CHECK(round_to(DType::FP16, 65504.0) == 65504.0, "fp16 max finite");
  CHECK(std::isinf(round_to(DType::FP16, 70000.0)), "fp16 overflow -> inf");
  CHECK(round_to(DType::FP16, 1.0 + 1.0 / 2048) == 1.0, "fp16 RNE tie to even (down)");
  CHECK(round_to(DType::FP16, 1.0 + 3.0 / 2048) == 1.0 + 2.0 / 1024, "fp16 RNE tie to even (up)");
  CHECK(round_to(DType::FP16, std::ldexp(1.0, -24)) == std::ldexp(1.0, -24), "fp16 min subnormal");
  CHECK(round_to(DType::BF16, 1.0 + 1.0 / 256) == 1.0, "bf16 RNE tie");
  CHECK(round_to(DType::BF16, 3.0e38) == 3.0e38 || std::isfinite(round_to(DType::BF16, 3.0e38)), "bf16 range");
  CHECK(round_to(DType::FP32, 0.1) == static_cast<double>(0.1f), "fp32 matches float cast");
}

static void test_order_matters_in_low_precision() {
  // Classic: large + many small. Sequential fp16 loses the small terms, pairwise keeps more.
  std::vector<double> v(1, 2048.0);
  for (int i = 0; i < 1024; ++i) v.push_back(1.0);
  const double seq = reduce_sum(v, DType::FP16, Order::Sequential, 32);
  const double pw = reduce_sum(v, DType::FP16, Order::Pairwise, 32);
  const double exact = reduce_sum(v, DType::FP64, Order::Sequential, 32);
  CHECK(exact == 3072.0, "fp64 exact sum");
  CHECK(seq != pw, "fp16 sequential vs pairwise differ");
  CHECK(std::fabs(pw - exact) < std::fabs(seq - exact), "pairwise closer to exact than sequential");
}

static void test_oracle_vs_fp32_close() {
  MoEConfig cfg;
  auto w = MoEWeights::random(cfg, 7);
  std::mt19937_64 rng(11);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> x(cfg.hidden);
  for (auto& v : x) v = nd(rng);
  const auto ref = forward(w, x.data(), Numerics::oracle());
  const auto f32 = forward(w, x.data(), Numerics{});
  double max_rel = 0.0, max_ref = 0.0;
  for (int d = 0; d < cfg.hidden; ++d) {
    max_rel = std::fmax(max_rel, std::fabs(ref.y[d] - f32.y[d]));
    max_ref = std::fmax(max_ref, std::fabs(ref.y[d]));
  }
  CHECK(max_rel / max_ref < 1e-5, "fp32 within 1e-5 of fp64 oracle");
  CHECK(topk_set_mismatch(ref.route.topk_ids, f32.route.topk_ids) == 0 || ref.route.margin < 1e-5,
        "fp32 routes match oracle unless near-tie");
}

static void test_deterministic_and_tiebreak() {
  MoEConfig cfg;
  cfg.n_experts = 8;
  cfg.top_k = 2;
  auto w = MoEWeights::random(cfg, 3);
  // Force an exact tie between experts 5 and 2 at the k-th slot: make their router rows equal.
  for (int c = 0; c < cfg.hidden; ++c) w.router[5 * cfg.hidden + c] = w.router[2 * cfg.hidden + c];
  std::vector<double> x(cfg.hidden, 0.1);
  const auto a = route(w, x.data(), Numerics::oracle());
  const auto b = route(w, x.data(), Numerics::oracle());
  CHECK(a.topk_ids == b.topk_ids, "same config is bit-deterministic");
  bool has2 = false, has5 = false;
  for (int id : a.topk_ids) { has2 |= id == 2; has5 |= id == 5; }
  CHECK(!(has5 && !has2), "tie broken toward lower index");
}

static void test_flip_census_smoke() {
  // How often does an fp16-storage/bf16-router config flip routes vs the oracle?
  MoEConfig cfg;
  cfg.n_experts = 64;
  cfg.top_k = 8;
  auto w = MoEWeights::random(cfg, 42);
  Numerics low;
  low.storage = DType::FP16;
  low.accum = DType::FP16;
  low.router_accum = DType::BF16;
  std::mt19937_64 rng(1);
  std::normal_distribution<double> nd(0.0, 1.0);
  int flips = 0, n = 2000;
  std::vector<double> x(cfg.hidden);
  for (int t = 0; t < n; ++t) {
    for (auto& v : x) v = nd(rng);
    const auto a = route(w, x.data(), Numerics::oracle());
    const auto b = route(w, x.data(), low);
    flips += topk_set_mismatch(a.topk_ids, b.topk_ids) > 0;
  }
  std::printf("  smoke: %d/%d tokens flipped (fp16 storage + bf16 router vs fp64)  [%s]\n", flips, n,
              low.describe().c_str());
  CHECK(flips > 0, "low precision router produces some flips on random data");
  CHECK(flips < n, "but not on every token");
}

int main() {
  test_rounding();
  test_order_matters_in_low_precision();
  test_oracle_vs_fp32_close();
  test_deterministic_and_tiebreak();
  test_flip_census_smoke();
  if (failures == 0) std::printf("ALL TESTS PASSED\n");
  return failures == 0 ? 0 : 1;
}
