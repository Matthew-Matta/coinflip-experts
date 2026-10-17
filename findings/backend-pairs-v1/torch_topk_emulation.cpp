// Emulates aten/src/ATen/native/cpu/SortingKernel.cpp topk_impl_loop (largest=true, sorted=true) with this
// toolchain's C++ standard library: value-only comparator; partial_sort if k*64 <= n, else nth_element(k-1) + sort.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>
#include "io.hpp"
using namespace coinflip;
int main(int argc, char** argv) {
  NpyArray p = npy_read(argv[1]), ids = npy_read(argv[2]);
  const size_t T = p.shape[0], n = p.shape[1], k = ids.shape[1];
  using elem = std::pair<float, long long>;
  auto cmp = [](const elem& x, const elem& y) { return (std::isnan(x.first) && !std::isnan(y.first)) || (x.first > y.first); };
  size_t exact_set = 0, exact_order = 0;
  std::vector<elem> q;
  for (size_t t = 0; t < T; ++t) {
    q.clear();
    for (size_t j = 0; j < n; ++j) q.emplace_back(static_cast<float>(p.data[t * n + j]), static_cast<long long>(j));
    if (k * 64 <= n) std::partial_sort(q.begin(), q.begin() + k, q.end(), cmp);
    else { std::nth_element(q.begin(), q.begin() + k - 1, q.end(), cmp); std::sort(q.begin(), q.begin() + k - 1, cmp); }
    std::vector<long long> mine(k), ref(k);
    for (size_t s = 0; s < k; ++s) { mine[s] = q[s].second; ref[s] = static_cast<long long>(ids.data[t * k + s]); }
    exact_order += mine == ref;
    std::sort(mine.begin(), mine.end()); std::sort(ref.begin(), ref.end());
    exact_set += mine == ref;
  }
  std::printf("tokens %zu: same top-k set %zu, same order %zu\n", T, exact_set, exact_order);
}
