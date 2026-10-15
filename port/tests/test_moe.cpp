// Self-contained tests (no framework). Exit code != 0 on failure.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "block.hpp"
#include "io.hpp"
#include "numerics.hpp"

using namespace coinflip;

static int failures = 0;
#define CHECK(cond, msg)                                         \
  do {                                                           \
    if (!(cond)) {                                               \
      std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);  \
      ++failures;                                                \
    }                                                            \
  } while (0)

static void test_rounding_known_values() {
  CHECK(round_to(DType::FP16, 1.0) == 1.0, "fp16 exact 1");
  CHECK(round_to(DType::FP16, 65504.0) == 65504.0, "fp16 max finite");
  CHECK(round_to(DType::FP16, 65519.0) == 65504.0, "fp16 just below the overflow midpoint stays finite");
  CHECK(std::isinf(round_to(DType::FP16, 65520.0)), "fp16 overflow midpoint rounds to inf (max is odd)");
  CHECK(round_to(DType::FP16, 1.0 + 1.0 / 2048) == 1.0, "fp16 tie to even (down)");
  CHECK(round_to(DType::FP16, 1.0 + 3.0 / 2048) == 1.0 + 2.0 / 1024, "fp16 tie to even (up)");
  CHECK(round_to(DType::FP16, std::ldexp(1.0, -24)) == std::ldexp(1.0, -24), "fp16 min subnormal");
  CHECK(round_to(DType::FP16, std::ldexp(1.0, -26)) == 0.0, "fp16 below half min subnormal -> 0");
  CHECK(round_to(DType::BF16, 1.0 + 1.0 / 256) == 1.0, "bf16 tie to even");
  const double bf16_max = std::ldexp(2.0 - std::ldexp(1.0, -7), 127);
  CHECK(round_to(DType::BF16, bf16_max) == bf16_max, "bf16 max finite");
  CHECK(std::isinf(round_to(DType::BF16, std::ldexp(2.0 - std::ldexp(1.0, -8), 127))), "bf16 overflow -> inf");
  CHECK(round_to(DType::FP32, 0.1) == static_cast<double>(0.1f), "fp32 matches float cast");
  CHECK(round_to(DType::BF16, -1.0 - 3.0 / 256) == -1.0 - 4.0 / 256, "bf16 negative tie to even");
  CHECK(ulp_of(DType::BF16, 1.0) == 1.0 / 128 && ulp_of(DType::FP16, 1.5) == 1.0 / 1024, "ulp_of");
}

static void test_fast_rounding_matches_slow() {
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> mant(1.0, 2.0);
  std::uniform_int_distribution<int> ex(-160, 140);
  int bad = 0;
  for (int i = 0; i < 1000000; ++i) {
    double v = std::ldexp(mant(rng), ex(rng)) * (i % 2 ? -1 : 1);
    if (i % 7 == 0) v = std::ldexp(1.0 + (2 * (i % 128) + 1) / 256.0, ex(rng) / 4);  // exact bf16 ties
    for (DType d : {DType::BF16, DType::FP16})
      if (round_to(d, v) != round_to_slow(d, v) && !(std::isnan(round_to(d, v)))) ++bad;
  }
  CHECK(bad == 0, "fast bf16/fp16 rounding == slow reference on 1e6 values");
}

static void test_orders() {
  std::vector<double> v(1, 2048.0);
  for (int i = 0; i < 1024; ++i) v.push_back(1.0);
  const double seq = reduce_sum(v, DType::FP16, Order::Sequential, 32);
  const double pw = reduce_sum(v, DType::FP16, Order::Pairwise, 32);
  CHECK(reduce_sum(v, DType::FP64, Order::Sequential, 32) == 3072.0, "fp64 exact sum");
  CHECK(seq != pw && std::fabs(pw - 3072.0) < std::fabs(seq - 3072.0), "fp16 pairwise beats sequential here");
  const std::vector<double> one{1.0 + 1.0 / 4096};
  CHECK(reduce_sum(one, DType::FP16, Order::Pairwise, 32) == 1.0, "pairwise singleton is rounded");
  CHECK(reduce_sum(v, DType::FP16, Order::Strided, 1) == seq, "strided with 1 lane == sequential");
}

static void test_fp32_fma_matches_hardware() {
  std::mt19937_64 rng(9);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n = 777;
  std::vector<float> w(n), xf(n);
  std::vector<double> x(n);
  for (int i = 0; i < n; ++i) { w[i] = nd(rng); xf[i] = nd(rng); x[i] = xf[i]; }
  float ref = 0.f, ref_nofma = 0.f;
  for (int i = 0; i < n; ++i) {
    ref = std::fmaf(w[i], xf[i], ref);
    const float p = w[i] * xf[i];
    ref_nofma = ref_nofma + p;
  }
  CHECK(dot(w.data(), x.data(), 0, n, {DType::FP32, Order::Sequential, 32, true}) == ref, "fp32 fma chain");
  CHECK(dot(w.data(), x.data(), 0, n, {DType::FP32, Order::Sequential, 32, false}) == ref_nofma, "fp32 no-fma chain");
}

static void test_collective() {
  std::mt19937_64 rng(3);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int rows = 64, cols = 512;
  std::vector<float> W(rows * cols);
  std::vector<double> x(cols), y1(rows), y2(rows), yr(rows), yt(rows);
  for (auto& v : W) v = static_cast<float>(round_to(DType::BF16, nd(rng) / 20));
  for (auto& v : x) v = round_to(DType::BF16, nd(rng));
  const AccSpec s{DType::FP32, Order::Sequential, 32, true};
  matvec(W.data(), rows, cols, x.data(), y1.data(), s, DType::FP64);
  matvec(W.data(), rows, cols, x.data(), y2.data(), s, DType::FP64, Collective{1, Topology::Tree, DType::BF16});
  CHECK(y1 == y2, "collective n=1 is a no-op");
  matvec(W.data(), rows, cols, x.data(), yr.data(), s, DType::FP64, Collective{8, Topology::Ring, DType::BF16});
  matvec(W.data(), rows, cols, x.data(), yt.data(), s, DType::FP64, Collective{8, Topology::Tree, DType::BF16});
  CHECK(yr != yt, "bf16 ring and tree differ at n=8");
}

static BlockConfig small_cfg() {
  BlockConfig c;
  c.hidden = 64;
  c.inter = 32;
  c.n_experts = 16;
  c.top_k = 4;
  return c;
}

static void inputs(int T, int H, uint64_t seed, std::vector<float>& oin, std::vector<float>& res) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  oin.resize(static_cast<size_t>(T) * H);
  res.resize(static_cast<size_t>(T) * H);
  for (auto& v : oin) v = static_cast<float>(round_to(DType::BF16, nd(rng)));
  for (auto& v : res) v = static_cast<float>(round_to(DType::BF16, nd(rng)));
}

// First op (in spec order) whose output differs bit-wise; "" if none.
static std::string first_diff(const Trace& a, const Trace& b) {
  for (const auto& op : op_order()) {
    if (op == "topk_ids") {
      if (a.ids.at(op) != b.ids.at(op)) return op;
    } else if (a.f.at(op) != b.f.at(op)) {
      return op;
    }
  }
  return "";
}

static void test_block_oracle_vs_fp32_and_determinism() {
  const auto w = BlockWeights::random(small_cfg(), 7);
  const int T = 64;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 11, oin, res);
  RunOptions o32;
  o32.num.storage = o32.num.gate_out = o32.num.weights = o32.num.combine_acc = DType::FP32;
  RunOptions oo;
  oo.num = Numerics::oracle();
  const Trace a = run_block(w, oin, res, T, o32), r = run_block(w, oin, res, T, oo);
  double num = 0, den = 0;
  for (size_t i = 0; i < r.f.at("output").size(); ++i) {
    num = std::fmax(num, std::fabs(a.f.at("output")[i] - r.f.at("output")[i]));
    den = std::fmax(den, std::fabs(r.f.at("output")[i]));
  }
  CHECK(num / den < 1e-5, "fp32 block within 1e-5 of fp64 oracle (max abs / max ref)");
  RunOptions single = o32;
  single.threads = 1;
  const Trace b = run_block(w, oin, res, T, single);
  CHECK(first_diff(a, b).empty(), "multi-threaded == single-threaded, bit for bit");
}

static void test_null_patch() {
  const auto w = BlockWeights::random(small_cfg(), 21);
  const int T = 32;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 4, oin, res);
  RunOptions base;
  const Trace clean = run_block(w, oin, res, T, base);
  for (const auto& op : op_order()) {
    RunOptions o;
    Patch p;
    p.op = op;
    if (op == "topk_ids") p.values.assign(clean.ids.at(op).begin(), clean.ids.at(op).end());
    else p.values = clean.f.at(op);
    p.ref_ids = clean.ids.at("topk_ids");
    o.patches.push_back(p);
    const Trace t = run_block(w, oin, res, T, o);
    const std::string msg = "null patch on " + op + " changes nothing";
    CHECK(first_diff(clean, t).empty() && t.patch_skipped_tokens == 0, msg.c_str());
  }
}

static void test_bugs_first_show_at_injected_op() {
  auto w = BlockWeights::random(small_cfg(), 33);
  const int T = 48;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 8, oin, res);
  RunOptions base;
  const Trace clean = run_block(w, oin, res, T, base);
  for (const auto& info : bug_registry()) {
    Bug b;
    b.type = info.type;
    if (info.type == "oproj_accum" || info.type == "rmsnorm_dtype" || info.type == "router_accum" ||
        info.type == "softmax_dtype" || info.type == "expert_accum")
      b.params["dtype"] = info.type == "expert_accum" || info.type == "oproj_accum" ? "fp16" : "bf16";
    if (info.type == "gate_out_dtype" || info.type == "weights_dtype") b.params["dtype"] = "fp32";
    if (info.type == "combine_accum") b.params["dtype"] = "fp32";
    if (info.type == "combine_order") b.params["order"] = "expert_desc";
    if (info.type == "expert_permuted") {
      b.params["a"] = std::to_string(clean.ids.at("topk_ids")[0]);
      b.params["b"] = std::to_string(clean.ids.at("topk_ids")[1]);
    }
    RunOptions o;
    o.bug = &b;
    const std::string got = first_diff(clean, run_block(w, oin, res, T, o));
    const bool ok = got == info.injected_op || (got.empty() && info.type == "topk_tiebreak_reversed");
    if (!ok) std::printf("  bug %s: first diff at '%s', registry says '%s'\n", info.type.c_str(), got.c_str(),
                         info.injected_op.c_str());
    CHECK(ok, ("bug " + info.type + " first shows at its injected op").c_str());
  }
}

static void test_npy_roundtrip_and_sha() {
  const std::string path = "/tmp/coinflip_test_roundtrip.npy";
  const std::vector<double> v{1.5, -2.25, 3.0, 0.0, 1e-3, 7.0};
  npy_write_f32(path, {2, 3}, v.data());
  const NpyArray a = npy_read(path);
  CHECK(a.descr == "<f4" && a.shape == std::vector<size_t>({2, 3}), "npy header round trip");
  bool same = true;
  for (size_t i = 0; i < v.size(); ++i) same &= a.data[i] == static_cast<double>(static_cast<float>(v[i]));
  CHECK(same, "npy data round trip");
  CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256('abc')");
  const Json j = json_parse(R"({"a": [1, 2.5, true], "b": {"c": "x\"y"}, "d": null})");
  CHECK(j.at("a").arr.size() == 3 && j.at("a").arr[1].num == 2.5 && j.at("b").at("c").str == "x\"y", "json parse");
}

int main() {
  test_rounding_known_values();
  test_fast_rounding_matches_slow();
  test_orders();
  test_fp32_fma_matches_hardware();
  test_collective();
  test_block_oracle_vs_fp32_and_determinism();
  test_null_patch();
  test_bugs_first_show_at_injected_op();
  test_npy_roundtrip_and_sha();
  if (failures == 0) std::printf("ALL TESTS PASSED\n");
  return failures == 0 ? 0 : 1;
}
