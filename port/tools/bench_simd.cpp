// bench_simd: time one fp32 matvec of OLMoE expert size, scalar (sequential FMA) vs SIMD, on the same data.
// Reports median over --runs timed repetitions (after warm-up), GFLOP/s and effective weight bandwidth (GB/s).
// Measure on an otherwise idle machine; the result is only claimed if it is measured (PREREG 3c).
// The weight working set decides what the GB/s figure means: 1024 x 2048 fp32 = 8 MB fits in the M1 Pro's 12 MB L2,
// so it measures cache bandwidth ("L2-resident"); 32768 x 2048 = 256 MB does not ("DRAM-bound"). Report both, labelled.
//   bench_simd [--rows 1024] [--cols 2048] [--runs 20] [--reps 50] [--accs 4] [--label L2-resident]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "numerics.hpp"

using namespace coinflip;

int main(int argc, char** argv) {
  int rows = 1024, cols = 2048, runs = 20, reps = 50, accs = 4;
  std::string label = "unlabelled";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i];
    if (a == "--label") { label = argv[i + 1]; continue; }
    const int v = std::stoi(argv[i + 1]);
    if (a == "--rows") rows = v; else if (a == "--cols") cols = v; else if (a == "--runs") runs = v;
    else if (a == "--reps") reps = v; else if (a == "--accs") accs = v;
    else { std::fprintf(stderr, "bench_simd: unknown flag %s\n", a.c_str()); return 2; }
  }
  if (argc % 2 == 0) { std::fprintf(stderr, "bench_simd: flags take a value\n"); return 2; }
  std::mt19937_64 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> W(static_cast<size_t>(rows) * cols);
  std::vector<double> x(cols), y(rows);
  for (auto& v : W) v = nd(rng);
  for (auto& v : x) v = nd(rng);
  AccSpec scalar{DType::FP32, Order::Sequential, 32, true};
  AccSpec simd = scalar;
  simd.simd = true;
  simd.simd_accs = accs;
  auto time_one = [&](const AccSpec& s) {
    for (int w = 0; w < 3; ++w) matvec(W.data(), rows, cols, x.data(), y.data(), s, DType::FP32);  // warm-up
    std::vector<double> t(runs);
    for (int r = 0; r < runs; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int k = 0; k < reps; ++k) matvec(W.data(), rows, cols, x.data(), y.data(), s, DType::FP32);
      t[r] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    std::sort(t.begin(), t.end());
    return std::make_pair(t[runs / 2], t);
  };
  const auto [ts, all_s] = time_one(scalar);
  const auto [tv, all_v] = time_one(simd);
  // Same order and 4 x accs accumulator lanes as the SIMD path, written as plain scalar C++ (the compiler may still vectorise
  // it). Separates "breaking the single FMA dependency chain" from "explicit NEON".
  std::vector<float> Wf(W), xf(x.begin(), x.end());
  std::vector<double> tp(runs);
  for (int w = 0; w < 3; ++w) for (int r = 0; r < rows; ++r) y[r] = dot_simd(Wf.data() + static_cast<size_t>(r) * cols, xf.data(), cols, accs, true);
  for (int rr = 0; rr < runs; ++rr) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; ++k)
      for (int r = 0; r < rows; ++r) y[r] = dot_simd(Wf.data() + static_cast<size_t>(r) * cols, xf.data(), cols, accs, true);
    tp[rr] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
  }
  std::sort(tp.begin(), tp.end());
  const double tport = tp[runs / 2];
  const double flops = 2.0 * rows * cols, bytes = 4.0 * rows * cols;
  std::printf("{\"label\": \"%s\", \"weight_MiB\": %.1f, \"rows\": %d, \"cols\": %d, \"runs\": %d, \"reps_per_run\": %d, \"simd_accs\": %d, \"simd_native\": %s,\n"
              " \"scalar_median_s\": %.6e, \"simd_median_s\": %.6e, \"speedup_median\": %.3f,\n"
              " \"scalar_gflops\": %.2f, \"simd_gflops\": %.2f, \"scalar_weight_GBps\": %.2f, \"simd_weight_GBps\": %.2f,\n"
              " \"scalar_min_max_s\": [%.6e, %.6e], \"simd_min_max_s\": [%.6e, %.6e],\n"
              " \"portable_same_order_median_s\": %.6e, \"speedup_simd_vs_portable\": %.3f, \"speedup_portable_vs_scalar\": %.3f,\n"
              " \"threads\": 1}\n",
              label.c_str(), bytes / (1024.0 * 1024.0), rows, cols, runs, reps, accs, simd_is_native() ? "true" : "false", ts, tv, ts / tv, flops / ts / 1e9,
              flops / tv / 1e9, bytes / ts / 1e9, bytes / tv / 1e9, all_s.front(), all_s.back(), all_v.front(), all_v.back(),
              tport, tport / tv, ts / tport);
  return 0;
}
