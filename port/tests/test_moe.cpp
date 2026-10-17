// Self-contained tests (no framework). Exit code != 0 on failure.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

// Qwen3-30B-A3B-shaped (scaled down): rectangular o_proj, renormalised top-k, more experts than OLMoE per top-k.
static BlockConfig small_qwen_cfg() {
  BlockConfig c = small_cfg();
  c.oproj_in = 96;
  c.n_experts = 32;
  c.top_k = 8;
  c.norm_topk_prob = true;
  c.rms_eps = 1e-6;
  return c;
}

static void inputs(int T, int H, uint64_t seed, std::vector<float>& oin, std::vector<float>& res, int Hin = 0) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  oin.resize(static_cast<size_t>(T) * (Hin > 0 ? Hin : H));
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

static void test_null_patch(const BlockConfig& cfg, const std::string& tag) {
  const auto w = BlockWeights::random(cfg, 21);
  const int T = 32;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 4, oin, res, oproj_width(cfg));
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
    const std::string msg = tag + ": null patch on " + op + " changes nothing";
    CHECK(first_diff(clean, t).empty() && t.patch_skipped_tokens == 0, msg.c_str());
  }
}

static void test_bugs_first_show_at_injected_op(const BlockConfig& cfg, const std::string& tag) {
  auto w = BlockWeights::random(cfg, 33);
  const int T = 48;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 8, oin, res, oproj_width(cfg));
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
    CHECK(ok, (tag + ": bug " + info.type + " first shows at its injected op").c_str());
  }
}

static void test_npy_roundtrip_and_sha() {
  const std::string path = "/tmp/coinflip_test_roundtrip_" + std::to_string(getpid()) + ".npy";
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
  // A header length larger than the file is an error, not an out-of-bounds read.
  {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const char bad[] = "\x93NUMPY\x01\x00\xff\x7f{'descr'";
    std::fwrite(bad, 1, sizeof bad - 1, f);
    std::fclose(f);
    bool threw = false;
    try { npy_read(path); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "npy header longer than the file is rejected");
  }
  std::remove(path.c_str());
}

// Order::TorchCascade: every element counted exactly once (exact in fp64 for integers) at sizes where the cascade
// levels engage, and a pinned fp32 value so a change to the order cannot pass silently.
static void test_torch_cascade() {
  bool exact = true;
  for (int n : {5, 255, 2048, 2051, 70001}) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) v[i] = i;
    exact &= reduce_sum(v, DType::FP64, Order::TorchCascade, 4) == 0.5 * n * (n - 1.0);
  }
  CHECK(exact, "torch_cascade sums every element exactly once");
  std::mt19937_64 rng(31);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> v(2048);
  for (auto& x : v) {
    const double y = round_to(DType::BF16, nd(rng));
    x = round_to(DType::FP32, y * y);
  }
  const double c = reduce_sum(v, DType::FP32, Order::TorchCascade, 4);
  CHECK(c == 0x1.00a6aap+11, "torch_cascade fp32 regression value");
  CHECK(c != reduce_sum(v, DType::FP32, Order::Sequential, 4), "torch_cascade differs from sequential here");
}

// A patched topk_ids array with an expert id outside [0, E) is an error.
static void test_patched_ids_are_range_checked() {
  const auto w = BlockWeights::random(small_cfg(), 21);
  const int T = 4;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 4, oin, res);
  RunOptions o;
  Patch p;
  p.op = "topk_ids";
  p.values.assign(static_cast<size_t>(T) * w.cfg.top_k, 0.0);
  for (int k = 0; k < w.cfg.top_k; ++k) p.values[k] = k;
  p.values[0] = w.cfg.n_experts;
  o.patches.push_back(p);
  bool threw = false;
  try { run_block(w, oin, res, T, o); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw, "out-of-range patched expert id is rejected");
}

static void test_simd_matches_documented_order() {
  std::mt19937_64 rng(17);
  std::normal_distribution<float> nd(0.f, 1.f);
  int bad = 0, differs_from_seq = 0;
  for (int n : {0, 3, 4, 15, 16, 17, 64, 1000, 1024, 2048, 2051}) {
    std::vector<float> w(n), x(n);
    std::vector<double> xd(n);
    for (int i = 0; i < n; ++i) { w[i] = nd(rng); x[i] = nd(rng); xd[i] = x[i]; }
    for (int accs : {1, 2, 4, 8}) {
      const float native = dot_simd(w.data(), x.data(), n, accs);
      const float portable = dot_simd(w.data(), x.data(), n, accs, /*force_portable=*/true);
      if (std::memcmp(&native, &portable, sizeof native) != 0) ++bad;
      if (n >= 64 && native != static_cast<float>(dot(w.data(), xd.data(), 0, n, {DType::FP32, Order::Sequential, 32, true})))
        ++differs_from_seq;
    }
  }
  CHECK(bad == 0, "SIMD (native) == portable loop in the documented order, bit for bit, incl. tails");
  CHECK(differs_from_seq > 0, "SIMD order genuinely differs from sequential FMA on some inputs");
  // 1 accumulator with no tail is exactly Order::Strided with 4 lanes and FMA.
  std::vector<float> w(1024), x(1024);
  std::vector<double> xd(1024);
  for (int i = 0; i < 1024; ++i) { w[i] = nd(rng); x[i] = nd(rng); xd[i] = x[i]; }
  CHECK(dot_simd(w.data(), x.data(), 1024, 1) ==
            static_cast<float>(dot(w.data(), xd.data(), 0, 1024, {DType::FP32, Order::Strided, 4, true})),
        "SIMD with one accumulator == Order::Strided(4) with FMA");
  // matvec with simd uses the same dot product.
  std::vector<double> y(1);
  AccSpec s{DType::FP32, Order::Sequential, 32, true};
  s.simd = true;
  matvec(w.data(), 1, 1024, xd.data(), y.data(), s, DType::FP64);
  CHECK(y[0] == static_cast<double>(dot_simd(w.data(), x.data(), 1024, 4)), "matvec(simd) == dot_simd");
  std::printf("  simd: native NEON %s\n", simd_is_native() ? "yes" : "no (portable loop)");
}

static void test_e8_formats_kahan_and_profile() {
  std::mt19937_64 rng(23);
  std::uniform_real_distribution<double> mant(1.0, 2.0);
  std::uniform_int_distribution<int> ex(-140, 127);
  int bad7 = 0, bad23 = 0;
  for (int i = 0; i < 200000; ++i) {
    const double v = (i & 1 ? -1 : 1) * std::ldexp(mant(rng), ex(rng));
    if (round_to(e8_format(7), v) != round_to(DType::BF16, v)) ++bad7;
    if (round_to(e8_format(23), v) != round_to(DType::FP32, v)) ++bad23;
  }
  CHECK(bad7 == 0 && bad23 == 0, "e8m7 rounds like bf16 and e8m23 like fp32 (incl. subnormals and overflow)");
  CHECK(parse_dtype("e8m10") == e8_format(10) && std::string(to_string(e8_format(10))) == "e8m10" &&
            mantissa_bits(e8_format(10)) == 10 && !is_e8_format(DType::FP32),
        "e8m<m> parse / name / mantissa bits");

  // A whole fp32 block with every accumulator in e8m23 (emulated path) is bit-identical to the native fp32 block.
  const auto w = BlockWeights::random(small_cfg(), 7);
  const int T = 32;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 11, oin, res);
  RunOptions base;
  base.num.storage = base.num.gate_out = base.num.weights = base.num.combine_acc = DType::FP32;
  RunOptions e23 = base;
  e23.num.oproj.acc = e23.num.router.acc = e23.num.expert.acc = e23.num.rms_acc = e23.num.combine_acc = e8_format(23);
  const Trace tb = run_block(w, oin, res, T, base), te = run_block(w, oin, res, T, e23);
  CHECK(first_diff(tb, te).empty(), "fp32 block with e8m23 accumulators == native fp32 block, bit for bit");
  const struct { const char* name; const char* first_op; } knobs[] = {
      {"oproj", "o_proj_out"}, {"rms", "router_input"}, {"router", "router_logits"}, {"expert", "expert_gate"},
      {"combine", "output"}};
  for (const auto& k : knobs) {
    RunOptions r = base;
    const DType m10 = e8_format(10);
    const std::string n = k.name;
    if (n == "oproj") r.num.oproj.acc = m10;
    if (n == "rms") r.num.rms_acc = m10;
    if (n == "router") r.num.router.acc = m10;
    if (n == "expert") r.num.expert.acc = m10;
    if (n == "combine") r.num.combine_acc = m10;
    CHECK(first_diff(tb, run_block(w, oin, res, T, r)) == k.first_op, ("e8m10 " + n + " first shows at its op").c_str());
  }

  // Kahan in a narrow accumulator recovers most of the precision a plain sequential sum loses.
  std::normal_distribution<double> nd(0.0, 1.0);
  const int n = 4096;
  std::vector<float> wv(n);
  std::vector<double> xv(n);
  double exact = 0;
  for (int i = 0; i < n; ++i) {
    wv[i] = static_cast<float>(round_to(DType::BF16, nd(rng)));
    xv[i] = round_to(DType::BF16, nd(rng));
    exact += static_cast<double>(wv[i]) * xv[i];
  }
  AccSpec plain{e8_format(10), Order::Sequential, 32, true}, comp = plain;
  comp.kahan = true;
  const double ep = std::fabs(dot(wv.data(), xv.data(), 0, n, plain) - exact);
  const double ek = std::fabs(dot(wv.data(), xv.data(), 0, n, comp) - exact);
  CHECK(ek * 8 < ep, "Kahan e8m10 dot is far closer to exact than plain e8m10");
  std::vector<double> ones(1000, 1.0 / 3.0);
  CHECK(std::fabs(kahan_sum(ones.data(), 1000, DType::BF16) - 1000.0 / 3.0) <
            std::fabs(reduce_sum(ones, DType::BF16, Order::Sequential, 32) - 1000.0 / 3.0),
        "Kahan bf16 sum beats sequential bf16");

  // Profile: one entry per stage, non-negative, and it does not change the result.
  Profile prof;
  RunOptions p = base;
  p.profile = &prof;
  p.threads = 1;
  const Trace tp = run_block(w, oin, res, T, p);
  double tot = 0;
  bool nonneg = true;
  for (double v : prof) { tot += v; nonneg = nonneg && v >= 0; }
  CHECK(prof.size() == profile_stages().size() && nonneg && tot > 0, "profile has every stage, all >= 0");
  CHECK(first_diff(tb, tp).empty(), "profiling does not change results");
}

// norm_topk_prob: weights are the chosen probabilities divided by their sum, and the sum order is configurable.
static void test_renorm() {
  const auto w = BlockWeights::random(small_qwen_cfg(), 5);
  const int T = 16, K = w.cfg.top_k;
  std::vector<float> oin, res;
  inputs(T, w.cfg.hidden, 9, oin, res, oproj_width(w.cfg));
  RunOptions o;
  o.num.storage = o.num.gate_out = o.num.weights = o.num.combine_acc = DType::FP32;
  const Trace t = run_block(w, oin, res, T, o);
  double worst = 0;
  for (int i = 0; i < T; ++i) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += t.f.at("topk_weights")[static_cast<size_t>(i) * K + k];
    worst = std::max(worst, std::fabs(s - 1.0));
  }
  CHECK(worst < 1e-6, "renormalised top-k weights sum to 1");
  RunOptions p = o;
  p.num.renorm_order = Order::Pairwise;
  CHECK(first_diff(t, run_block(w, oin, res, T, p)) == "topk_weights", "renorm order first changes topk_weights");
}

int main() {
  test_rounding_known_values();
  test_fast_rounding_matches_slow();
  test_orders();
  test_fp32_fma_matches_hardware();
  test_collective();
  test_block_oracle_vs_fp32_and_determinism();
  test_null_patch(small_cfg(), "olmoe-shaped");
  test_bugs_first_show_at_injected_op(small_cfg(), "olmoe-shaped");
  test_null_patch(small_qwen_cfg(), "qwen-shaped");
  test_bugs_first_show_at_injected_op(small_qwen_cfg(), "qwen-shaped");
  test_renorm();
  test_npy_roundtrip_and_sha();
  test_torch_cascade();
  test_patched_ids_are_range_checked();
  test_simd_matches_documented_order();
  test_e8_formats_kahan_and_profile();
  if (failures == 0) std::printf("ALL TESTS PASSED\n");
  return failures == 0 ? 0 : 1;
}
