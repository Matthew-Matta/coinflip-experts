#include "block.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace coinflip {

const char* to_string(CombineOrder o) {
  switch (o) {
    case CombineOrder::ExpertAsc: return "expert_asc";
    case CombineOrder::ExpertDesc: return "expert_desc";
    case CombineOrder::Slot: return "slot";
  }
  return "?";
}

CombineOrder parse_combine_order(const std::string& s) {
  if (s == "expert_asc") return CombineOrder::ExpertAsc;
  if (s == "expert_desc") return CombineOrder::ExpertDesc;
  if (s == "slot") return CombineOrder::Slot;
  throw std::invalid_argument("unknown combine order: " + s);
}

CollectiveOn parse_collective_on(const std::string& s) {
  if (s == "none") return CollectiveOn::None;
  if (s == "oproj") return CollectiveOn::OProj;
  if (s == "down") return CollectiveOn::Down;
  if (s == "both") return CollectiveOn::Both;
  throw std::invalid_argument("unknown collective target: " + s);
}

Numerics Numerics::oracle() {
  Numerics n;
  const AccSpec exact{DType::FP64, Order::Sequential, 32, true};
  n.storage = n.rms = n.gate_out = n.softmax = n.weights = n.combine_acc = DType::FP64;
  n.oproj = n.router = n.expert = exact;
  return n;
}

const std::vector<std::string>& op_order() {
  static const std::vector<std::string> ops = {
      "o_proj_out", "resid_sum",   "router_input", "router_logits", "router_probs",  "topk_ids",  "topk_weights",
      "expert_gate", "expert_up", "expert_act",   "expert_down",   "combine_terms", "output"};
  return ops;
}

const std::vector<BugInfo>& bug_registry() {
  static const std::vector<BugInfo> reg = {
      {"oproj_accum", "o_proj_out", "o_proj accumulator dtype (params: dtype)"},
      {"rmsnorm_dtype", "router_input", "RMSNorm internal precision (params: dtype)"},
      {"router_accum", "router_logits", "router matmul accumulator dtype (params: dtype)"},
      {"gate_out_dtype", "router_logits", "dtype the router logits are stored in (params: dtype)"},
      {"softmax_dtype", "router_probs", "softmax precision (params: dtype)"},
      {"topk_tiebreak_reversed", "topk_ids", "exact ties go to the higher expert id"},
      {"topk_renorm_toggle", "topk_weights", "top-k renormalisation flipped relative to the config"},
      {"weights_dtype", "topk_weights", "dtype routing weights are cast to before combine (params: dtype)"},
      {"expert_accum", "expert_gate", "expert matmul accumulator dtype (params: dtype)"},
      {"gate_up_swapped", "expert_gate", "gate and up halves of gate_up_proj swapped at load"},
      {"expert_permuted", "expert_gate", "weights of experts a and b swapped at load (params: a, b)"},
      {"combine_accum", "output", "combine accumulator dtype (params: dtype)"},
      {"combine_order", "output", "combine summation order (params: order = expert_desc | slot)"},
  };
  return reg;
}

int topk_set_mismatch(const int* a, const int* b, int k) {
  int miss = 0;
  for (int i = 0; i < k; ++i)
    if (std::find(b, b + k, a[i]) == b + k) ++miss;
  return miss;
}

BlockWeights BlockWeights::random(const BlockConfig& cfg, uint64_t seed) {
  BlockWeights w;
  w.cfg = cfg;
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int H = cfg.hidden, I = cfg.inter, E = cfg.n_experts;
  auto fill = [&](std::vector<float>& v, size_t n, double scale) {
    v.resize(n);
    for (auto& x : v) x = static_cast<float>(round_to(DType::BF16, nd(rng) * scale));
  };
  fill(w.o_proj, static_cast<size_t>(H) * H, 1.0 / std::sqrt(H));
  w.norm.resize(H);
  for (auto& x : w.norm) x = static_cast<float>(round_to(DType::BF16, 1.0 + 0.1 * nd(rng)));
  fill(w.router, static_cast<size_t>(E) * H, 1.0 / std::sqrt(H));
  fill(w.gate_up, static_cast<size_t>(E) * 2 * I * H, 1.0 / std::sqrt(H));
  fill(w.down, static_cast<size_t>(E) * H * I, 1.0 / std::sqrt(I));
  return w;
}

namespace {

std::string param(const Bug& b, const std::string& k) {
  auto it = b.params.find(k);
  if (it == b.params.end()) throw std::invalid_argument("bug " + b.type + " needs param '" + k + "'");
  return it->second;
}

struct Effective {
  Numerics num;
  bool norm_topk = false;
  bool gate_up_swapped = false;
  std::vector<int> expert_map;  // logical expert -> weight slot
};

Effective apply_bug(const BlockConfig& cfg, const Numerics& base, const Bug* bug) {
  Effective e{base, cfg.norm_topk_prob, false, {}};
  e.expert_map.resize(cfg.n_experts);
  std::iota(e.expert_map.begin(), e.expert_map.end(), 0);
  if (!bug) return e;
  const auto& reg = bug_registry();
  if (std::none_of(reg.begin(), reg.end(), [&](const BugInfo& i) { return i.type == bug->type; }))
    throw std::invalid_argument("unknown bug type: " + bug->type);
  const std::string& t = bug->type;
  Numerics& n = e.num;
  if (t == "oproj_accum") n.oproj.acc = parse_dtype(param(*bug, "dtype"));
  else if (t == "rmsnorm_dtype") n.rms = parse_dtype(param(*bug, "dtype"));
  else if (t == "router_accum") n.router.acc = parse_dtype(param(*bug, "dtype"));
  else if (t == "gate_out_dtype") n.gate_out = parse_dtype(param(*bug, "dtype"));
  else if (t == "softmax_dtype") n.softmax = parse_dtype(param(*bug, "dtype"));
  else if (t == "topk_tiebreak_reversed") n.tiebreak_lower_index = false;
  else if (t == "topk_renorm_toggle") e.norm_topk = !e.norm_topk;
  else if (t == "weights_dtype") n.weights = parse_dtype(param(*bug, "dtype"));
  else if (t == "expert_accum") n.expert.acc = parse_dtype(param(*bug, "dtype"));
  else if (t == "gate_up_swapped") e.gate_up_swapped = true;
  else if (t == "expert_permuted") {
    const int a = std::stoi(param(*bug, "a")), b = std::stoi(param(*bug, "b"));
    if (a < 0 || b < 0 || a >= cfg.n_experts || b >= cfg.n_experts) throw std::invalid_argument("expert id out of range");
    std::swap(e.expert_map[a], e.expert_map[b]);
  } else if (t == "combine_accum") n.combine_acc = parse_dtype(param(*bug, "dtype"));
  else if (t == "combine_order") n.combine_order = parse_combine_order(param(*bug, "order"));
  return e;
}

struct OpShape {
  std::vector<int> dims;
  bool selection;
  bool slot_aligned;
};

std::map<std::string, OpShape> op_shapes(const BlockConfig& c) {
  const int H = c.hidden, I = c.inter, E = c.n_experts, K = c.top_k;
  return {
      {"o_proj_out", {{H}, true, false}},     {"resid_sum", {{H}, true, false}},
      {"router_input", {{H}, true, false}},   {"router_logits", {{E}, true, false}},
      {"router_probs", {{E}, true, false}},   {"topk_ids", {{K}, true, false}},
      {"topk_weights", {{K}, false, true}},   {"expert_gate", {{K, I}, false, true}},
      {"expert_up", {{K, I}, false, true}},   {"expert_act", {{K, I}, false, true}},
      {"expert_down", {{K, H}, false, true}}, {"combine_terms", {{K, H}, false, true}},
      {"output", {{H}, false, false}},
  };
}

size_t per_token(const OpShape& s) {
  size_t n = 1;
  for (int d : s.dims) n *= static_cast<size_t>(d);
  return n;
}

}  // namespace

Trace run_block(const BlockWeights& w, const std::vector<float>& o_proj_in, const std::vector<float>& residual,
                int T, const RunOptions& opt) {
  const BlockConfig& cfg = w.cfg;
  const int H = cfg.hidden, I = cfg.inter, E = cfg.n_experts, K = cfg.top_k;
  if (K >= E) throw std::invalid_argument("top_k must be < n_experts");
  if (o_proj_in.size() != static_cast<size_t>(T) * H || residual.size() != static_cast<size_t>(T) * H)
    throw std::invalid_argument("input shapes do not match [T, H]");

  const Effective eff = apply_bug(cfg, opt.num, opt.bug);
  const Numerics& num = eff.num;
  const DType S = num.storage;
  const DType silu_prec = S == DType::FP64 ? DType::FP64 : DType::FP32;
  const bool coll_oproj = num.collective_on == CollectiveOn::OProj || num.collective_on == CollectiveOn::Both;
  const bool coll_down = num.collective_on == CollectiveOn::Down || num.collective_on == CollectiveOn::Both;

  const auto shapes = op_shapes(cfg);
  std::map<std::string, const Patch*> patch;
  for (const auto& p : opt.patches) {
    auto it = shapes.find(p.op);
    if (it == shapes.end()) throw std::invalid_argument("cannot patch unknown op: " + p.op);
    if (p.values.size() != static_cast<size_t>(T) * per_token(it->second))
      throw std::invalid_argument("patch for " + p.op + " has the wrong size");
    if (it->second.slot_aligned && p.ref_ids.size() != static_cast<size_t>(T) * K)
      throw std::invalid_argument("patch for slot-aligned op " + p.op + " needs ref_ids [T, K]");
    patch[p.op] = &p;
  }

  Trace tr;
  tr.tokens = T;
  for (const auto& op : op_order()) {
    const auto& s = shapes.at(op);
    tr.shape[op] = s.dims;
    tr.stage[op] = s.selection ? "selection" : "post-selection";
    if (op == "topk_ids") tr.ids[op].assign(static_cast<size_t>(T) * K, 0);
    else tr.f[op].assign(static_cast<size_t>(T) * per_token(s), 0.0);
    tr.logical[op] = op == "router_probs" ? num.softmax : op == "router_logits" ? num.gate_out
                     : op == "topk_weights" ? num.weights : S;
  }
  std::atomic<int> skipped{0};

  auto token = [&](int t) {
    auto row = [&](const std::string& op) { return tr.f.at(op).data() + static_cast<size_t>(t) * per_token(shapes.at(op)); };
    auto dense_patch = [&](const std::string& op, double* v) {
      auto it = patch.find(op);
      if (it == patch.end()) return;
      const size_t n = per_token(shapes.at(op));
      std::copy_n(it->second->values.data() + static_cast<size_t>(t) * n, n, v);
    };
    // Slot-aligned patch: copy the source row for the same expert id; only if the top-k sets match.
    auto slot_patch = [&](const std::string& op, double* v, const int* ids, int width) {
      auto it = patch.find(op);
      if (it == patch.end()) return;
      const int* ref = it->second->ref_ids.data() + static_cast<size_t>(t) * K;
      if (topk_set_mismatch(ids, ref, K) != 0) {
        ++skipped;
        return;
      }
      for (int s = 0; s < K; ++s) {
        const int src = static_cast<int>(std::find(ref, ref + K, ids[s]) - ref);
        std::copy_n(it->second->values.data() + (static_cast<size_t>(t) * K + src) * width, width,
                    v + static_cast<size_t>(s) * width);
      }
    };

    // 1-2. o_proj (optionally through an emulated all-reduce), residual add.
    std::vector<double> oin(H), res(H);
    for (int i = 0; i < H; ++i) {
      oin[i] = round_to(S, o_proj_in[static_cast<size_t>(t) * H + i]);
      res[i] = round_to(S, residual[static_cast<size_t>(t) * H + i]);
    }
    double* oo = row("o_proj_out");
    matvec(w.o_proj.data(), H, H, oin.data(), oo, num.oproj, S, coll_oproj ? num.collective : Collective{});
    dense_patch("o_proj_out", oo);
    double* rs = row("resid_sum");
    for (int i = 0; i < H; ++i) rs[i] = round_to(S, res[i] + oo[i]);
    dense_patch("resid_sum", rs);

    // 3. RMSNorm: internal precision num.rms, cast to storage, then weight multiply in storage (HF :58-63).
    double* ri = row("router_input");
    {
      const DType R = num.rms;
      double ss = 0.0;
      std::vector<double> xr(H);
      for (int i = 0; i < H; ++i) {
        xr[i] = round_to(R, rs[i]);
        ss = round_to(R, ss + round_to(R, xr[i] * xr[i]));
      }
      const double var = round_to(R, ss / H);
      const double inv = round_to(R, 1.0 / std::sqrt(round_to(R, var + cfg.rms_eps)));
      for (int i = 0; i < H; ++i) ri[i] = round_to(S, w.norm[i] * round_to(S, round_to(R, xr[i] * inv)));
    }
    dense_patch("router_input", ri);

    // 4. Router logits: accumulate with num.router, store as num.gate_out.
    double* lg = row("router_logits");
    matvec(w.router.data(), E, H, ri, lg, num.router, num.gate_out);
    dense_patch("router_logits", lg);

    // 5. Softmax in num.softmax (max-subtracted).
    double* pr = row("router_probs");
    {
      const DType P = num.softmax;
      double mx = -INFINITY;
      for (int e = 0; e < E; ++e) mx = std::max(mx, round_to(P, lg[e]));
      double z = 0.0;
      for (int e = 0; e < E; ++e) {
        pr[e] = round_to(P, std::exp(round_to(P, round_to(P, lg[e]) - mx)));
        z = round_to(P, z + pr[e]);
      }
      for (int e = 0; e < E; ++e) pr[e] = round_to(P, pr[e] / z);
    }
    dense_patch("router_probs", pr);

    // 6. Top-k, sorted by descending key; exact ties by expert id.
    int* ids = tr.ids.at("topk_ids").data() + static_cast<size_t>(t) * K;
    {
      const double* key = num.topk_on_logits ? lg : pr;
      std::vector<int> idx(E);
      std::iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (key[a] != key[b]) return key[a] > key[b];
        return num.tiebreak_lower_index ? a < b : a > b;
      });
      std::copy_n(idx.begin(), K, ids);
      auto it = patch.find("topk_ids");
      if (it != patch.end())
        for (int s = 0; s < K; ++s) ids[s] = static_cast<int>(it->second->values[static_cast<size_t>(t) * K + s]);
    }

    // 7. Routing weights: probabilities of the chosen experts (renormalised if configured), then cast.
    double* tw = row("topk_weights");
    {
      double sum = 0.0;
      for (int s = 0; s < K; ++s) {
        tw[s] = pr[ids[s]];
        sum = round_to(num.softmax, sum + tw[s]);
      }
      for (int s = 0; s < K; ++s) {
        if (eff.norm_topk) tw[s] = round_to(num.softmax, tw[s] / sum);
        tw[s] = round_to(num.weights, tw[s]);
      }
    }
    slot_patch("topk_weights", tw, ids, 1);

    // 8-11. SwiGLU experts per slot (HF :330-334).
    double* g = row("expert_gate");
    double* u = row("expert_up");
    double* a = row("expert_act");
    double* d = row("expert_down");
    for (int s = 0; s < K; ++s) {
      const int ew = eff.expert_map[ids[s]];
      const float* gate_w = w.gate_up.data() + static_cast<size_t>(ew) * 2 * I * H;
      const float* up_w = gate_w + static_cast<size_t>(I) * H;
      if (eff.gate_up_swapped) std::swap(gate_w, up_w);
      matvec(gate_w, I, H, ri, g + static_cast<size_t>(s) * I, num.expert, S);
      matvec(up_w, I, H, ri, u + static_cast<size_t>(s) * I, num.expert, S);
    }
    slot_patch("expert_gate", g, ids, I);
    slot_patch("expert_up", u, ids, I);
    for (int i = 0; i < K * I; ++i) {
      const double silu = round_to(S, round_to(silu_prec, g[i] / (1.0 + std::exp(-g[i]))));
      a[i] = round_to(S, silu * u[i]);
    }
    slot_patch("expert_act", a, ids, I);
    for (int s = 0; s < K; ++s) {
      const int ew = eff.expert_map[ids[s]];
      matvec(w.down.data() + static_cast<size_t>(ew) * H * I, H, I, a + static_cast<size_t>(s) * I,
             d + static_cast<size_t>(s) * H, num.expert, S, coll_down ? num.collective : Collective{});
    }
    slot_patch("expert_down", d, ids, H);

    // 12. Combine terms (HF :335) and 13. combine (HF :336: index_add_ into a model-dtype buffer).
    double* ct = row("combine_terms");
    for (int s = 0; s < K; ++s)
      for (int h = 0; h < H; ++h) ct[static_cast<size_t>(s) * H + h] = round_to(S, d[static_cast<size_t>(s) * H + h] * tw[s]);
    slot_patch("combine_terms", ct, ids, H);

    std::vector<int> order(K);
    std::iota(order.begin(), order.end(), 0);
    if (num.combine_order != CombineOrder::Slot)
      std::sort(order.begin(), order.end(), [&](int x, int y) {
        return num.combine_order == CombineOrder::ExpertAsc ? ids[x] < ids[y] : ids[x] > ids[y];
      });
    double* out = row("output");
    for (int h = 0; h < H; ++h) {
      double acc = 0.0;
      for (int s : order) acc = round_to(num.combine_acc, acc + ct[static_cast<size_t>(s) * H + h]);
      out[h] = round_to(S, acc);
    }
    dense_patch("output", out);
  };

  int nthreads = opt.threads > 0 ? opt.threads : static_cast<int>(std::thread::hardware_concurrency());
  nthreads = std::max(1, std::min(nthreads, T));
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::atomic<bool> failed{false};
  for (int k = 0; k < nthreads; ++k)
    pool.emplace_back([&] {
      for (int t; !failed && (t = next++) < T;) {
        try {
          token(t);
        } catch (...) {
          if (!failed.exchange(true)) err = std::current_exception();
        }
      }
    });
  for (auto& th : pool) th.join();
  if (err) std::rethrow_exception(err);
  tr.patch_skipped_tokens = skipped;
  return tr;
}

}  // namespace coinflip
