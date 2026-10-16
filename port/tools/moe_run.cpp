// moe_run: run the C++ MoE block on an exported layer and write per-op dumps (spec/INTERFACES.md sections 1-3).
//
//   moe_run --export DIR --numerics numerics.json --dump OUT_DIR
//           [--bug bug.json] [--patch OP=path.npy ...] [--patch-ids topk_ids.npy] [--tokens a:b] [--threads N]
//   moe_run --list-bugs      (bug registry as JSON: type, injected_op, description)
//   --ops a,b,c writes only the listed ops (manifest lists only those); default writes all 13.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "block.hpp"
#include "io.hpp"

using namespace coinflip;
namespace fs = std::filesystem;

namespace {

void usage() {
  std::cerr << "usage: moe_run --export DIR --numerics FILE --dump OUT_DIR [--bug FILE] [--patch OP=FILE ...]\n"
               "               [--patch-ids FILE] [--tokens a:b] [--threads N]\n";
}

std::string cmd_output(const std::string& cmd) {
  std::string out;
  if (FILE* p = popen(cmd.c_str(), "r")) {
    char buf[256];
    while (fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
  }
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

// Numerics presets, then flat overrides. Unknown keys are an error.
Numerics load_numerics(const Json& j, std::string* preset_out) {
  Numerics n;
  const std::string preset = j.has("preset") ? j.at("preset").as_string() : "hf_bf16";
  if (preset == "oracle") {
    n = Numerics::oracle();
  } else if (preset == "hf_fp32") {
    n.storage = n.gate_out = n.weights = n.combine_acc = DType::FP32;
  } else if (preset != "hf_bf16") {
    throw std::invalid_argument("unknown numerics preset: " + preset);
  }
  *preset_out = preset;
  auto acc = [&](AccSpec& a, const std::string& p, const std::string& key, const Json& v) {
    if (key == p + "_accum") a.acc = parse_dtype(v.as_string());
    else if (key == p + "_order") a.order = parse_order(v.as_string());
    else if (key == p + "_block") a.block = std::stoi(v.as_string());
    else if (key == p + "_fma") a.fma = v.as_string() == "true";
    else return false;
    return true;
  };
  for (const auto& [k, v] : j.obj) {
    if (k == "preset" || k == "comment") continue;
    if (acc(n.oproj, "oproj", k, v) || acc(n.router, "router", k, v) || acc(n.expert, "expert", k, v)) continue;
    if (k == "storage") n.storage = parse_dtype(v.as_string());
    else if (k == "rms") n.rms = parse_dtype(v.as_string());
    else if (k == "rms_order") n.rms_order = parse_order(v.as_string());
    else if (k == "rms_block") n.rms_block = std::stoi(v.as_string());
    else if (k == "gate_out") n.gate_out = parse_dtype(v.as_string());
    else if (k == "softmax") n.softmax = parse_dtype(v.as_string());
    else if (k == "topk_on") n.topk_on_logits = v.as_string() == "logits";
    else if (k == "tiebreak") n.tiebreak_lower_index = v.as_string() != "higher";
    else if (k == "weights") n.weights = parse_dtype(v.as_string());
    else if (k == "combine_accum") n.combine_acc = parse_dtype(v.as_string());
    else if (k == "combine_order") n.combine_order = parse_combine_order(v.as_string());
    else if (k == "collective_n") n.collective.n = std::stoi(v.as_string());
    else if (k == "collective_topology") n.collective.topo = parse_topology(v.as_string());
    else if (k == "collective_wire") n.collective.wire = parse_dtype(v.as_string());
    else if (k == "collective_on") n.collective_on = parse_collective_on(v.as_string());
    else throw std::invalid_argument("unknown numerics key: " + k);
  }
  return n;
}

std::string describe(const AccSpec& a) {
  std::ostringstream s;
  s << "{\"accum\": \"" << to_string(a.acc) << "\", \"order\": \"" << to_string(a.order) << "\", \"block\": " << a.block
    << ", \"fma\": " << (a.fma ? "true" : "false") << "}";
  return s.str();
}

std::string numerics_json(const Numerics& n, const std::string& preset) {
  const char* on[] = {"none", "oproj", "down", "both"};
  std::ostringstream s;
  s << "{\"preset\": \"" << preset << "\", \"storage\": \"" << to_string(n.storage) << "\", \"oproj\": " << describe(n.oproj)
    << ", \"rms\": \"" << to_string(n.rms) << "\", \"rms_order\": \"" << to_string(n.rms_order) << "\", \"rms_block\": " << n.rms_block << ", \"router\": " << describe(n.router) << ", \"gate_out\": \""
    << to_string(n.gate_out) << "\", \"softmax\": \"" << to_string(n.softmax) << "\", \"topk_on\": \""
    << (n.topk_on_logits ? "logits" : "probs") << "\", \"tiebreak\": \"" << (n.tiebreak_lower_index ? "lower" : "higher")
    << "\", \"weights\": \"" << to_string(n.weights) << "\", \"expert\": " << describe(n.expert) << ", \"combine_accum\": \""
    << to_string(n.combine_acc) << "\", \"combine_order\": \"" << to_string(n.combine_order) << "\", \"collective\": {\"n\": "
    << n.collective.n << ", \"topology\": \"" << to_string(n.collective.topo) << "\", \"wire\": \"" << to_string(n.collective.wire)
    << "\", \"on\": \"" << on[static_cast<int>(n.collective_on)] << "\"}}";
  return s.str();
}

std::vector<float> slice_rows(const std::vector<float>& v, int width, int a, int b) {
  return std::vector<float>(v.begin() + static_cast<long>(a) * width, v.begin() + static_cast<long>(b) * width);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string export_dir, numerics_path, dump_dir, bug_path, patch_ids_path, token_range, ops_arg;
    std::vector<std::pair<std::string, std::string>> patch_args;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--export") export_dir = next();
      else if (a == "--numerics") numerics_path = next();
      else if (a == "--dump") dump_dir = next();
      else if (a == "--bug") bug_path = next();
      else if (a == "--patch-ids") patch_ids_path = next();
      else if (a == "--tokens") token_range = next();
      else if (a == "--threads") threads = std::stoi(next());
      else if (a == "--ops") ops_arg = next();
      else if (a == "--patch") {
        const std::string v = next();
        const auto eq = v.find('=');
        if (eq == std::string::npos) throw std::invalid_argument("--patch expects OP=path.npy");
        patch_args.emplace_back(v.substr(0, eq), v.substr(eq + 1));
      } else if (a == "--list-bugs") {
        std::cout << "[\n";
        const auto& reg = bug_registry();
        for (size_t k = 0; k < reg.size(); ++k)
          std::cout << "  {\"type\": \"" << reg[k].type << "\", \"injected_op\": \"" << reg[k].injected_op
                    << "\", \"description\": \"" << json_escape(reg[k].description) << "\"}"
                    << (k + 1 < reg.size() ? ",\n" : "\n");
        std::cout << "]\n";
        return 0;
      } else if (a == "-h" || a == "--help") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("unknown argument: " + a);
      }
    }
    if (export_dir.empty() || numerics_path.empty() || dump_dir.empty()) {
      usage();
      return 2;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const Json cfgj = json_read_file(export_dir + "/config.json");
    BlockWeights w;
    BlockConfig& c = w.cfg;
    c.hidden = static_cast<int>(cfgj.at("hidden_size").num);
    c.inter = static_cast<int>(cfgj.at("intermediate_size").num);
    c.n_experts = static_cast<int>(cfgj.at("num_experts").num);
    c.top_k = static_cast<int>(cfgj.at("top_k").num);
    c.norm_topk_prob = cfgj.at("norm_topk_prob").b;
    c.rms_eps = cfgj.at("rms_norm_eps").num;
    const size_t H = c.hidden, I = c.inter, E = c.n_experts;
    const std::string wd = export_dir + "/weights/";
    w.o_proj = npy_read_f32(wd + "o_proj.npy", {H, H});
    w.norm = npy_read_f32(wd + "post_attention_layernorm.npy", {H});
    w.router = npy_read_f32(wd + "router.npy", {E, H});
    w.gate_up = npy_read_f32(wd + "gate_up_proj.npy", {E, 2 * I, H});
    w.down = npy_read_f32(wd + "down_proj.npy", {E, H, I});

    auto oin = npy_read_f32(export_dir + "/inputs/o_proj_in.npy", {});
    auto res = npy_read_f32(export_dir + "/inputs/residual.npy", {});
    if (oin.size() % H || res.size() != oin.size()) throw std::runtime_error("inputs are not [T, H]");
    int T_all = static_cast<int>(oin.size() / H), a = 0, b = T_all;
    if (!token_range.empty()) {
      const auto colon = token_range.find(':');
      a = std::stoi(token_range.substr(0, colon));
      b = std::stoi(token_range.substr(colon + 1));
      if (a < 0 || b > T_all || a >= b) throw std::invalid_argument("bad --tokens range");
      oin = slice_rows(oin, c.hidden, a, b);
      res = slice_rows(res, c.hidden, a, b);
    }
    const int T = b - a;

    RunOptions opt;
    std::string preset;
    opt.num = load_numerics(json_read_file(numerics_path), &preset);
    opt.threads = threads;
    Bug bug;
    std::string bug_sha;
    if (!bug_path.empty()) {
      const Json bj = json_read_file(bug_path);
      bug.id = bj.has("bug_id") ? bj.at("bug_id").as_string() : "";
      bug.type = bj.at("type").as_string();
      if (bj.has("params"))
        for (const auto& [k, v] : bj.at("params").obj) bug.params[k] = v.as_string();
      bug_sha = sha256_file(bug_path);
      opt.bug = &bug;
    }
    std::vector<int> ref_ids;
    if (!patch_ids_path.empty()) {
      const NpyArray p = npy_read(patch_ids_path);
      if (p.size() != static_cast<size_t>(T) * c.top_k) throw std::runtime_error("--patch-ids is not [T, K]");
      for (double v : p.data) ref_ids.push_back(static_cast<int>(v));
    }
    for (const auto& [op, path] : patch_args) {
      const NpyArray p = npy_read(path);
      Patch pt;
      pt.op = op;
      pt.values = p.data;
      pt.ref_ids = ref_ids;
      opt.patches.push_back(std::move(pt));
    }

    const Trace tr = run_block(w, oin, res, T, opt);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    fs::create_directories(dump_dir);
    std::ostringstream ops;
    bool first = true;
    std::set<std::string> keep;
    for (std::stringstream ss(ops_arg); ss.good();) {
      std::string op;
      std::getline(ss, op, ',');
      if (op.empty()) continue;
      if (std::find(op_order().begin(), op_order().end(), op) == op_order().end())
        throw std::invalid_argument("--ops: unknown op " + op);
      keep.insert(op);
    }
    size_t written = 0;
    for (const auto& op : op_order()) {
      if (!keep.empty() && !keep.count(op)) continue;
      ++written;
      std::vector<size_t> shape{static_cast<size_t>(T)};
      for (int d : tr.shape.at(op)) shape.push_back(static_cast<size_t>(d));
      const std::string file = op + ".npy";
      if (op == "topk_ids") npy_write_i32(dump_dir + "/" + file, shape, tr.ids.at(op).data());
      else npy_write_f32(dump_dir + "/" + file, shape, tr.f.at(op).data());
      ops << (first ? "" : ",\n    ") << "{\"name\": \"" << op << "\", \"file\": \"" << file << "\", \"shape\": [";
      for (size_t i = 0; i < shape.size(); ++i) ops << (i ? ", " : "") << shape[i];
      ops << "], \"logical_dtype\": \"" << (op == "topk_ids" ? "int32" : to_string(tr.logical.at(op)))
          << "\", \"stage\": \"" << tr.stage.at(op) << "\"}";
      first = false;
    }
    std::ostringstream patches;
    for (size_t i = 0; i < patch_args.size(); ++i)
      patches << (i ? ", " : "") << "{\"op\": \"" << patch_args[i].first << "\", \"sha256\": \""
              << sha256_file(patch_args[i].second) << "\"}";

    char ts[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    const std::string git_sha = cmd_output("git rev-parse HEAD 2>/dev/null");
    const std::string git_dirty = cmd_output("git status --porcelain 2>/dev/null | head -1");

    std::ofstream m(dump_dir + "/manifest.json");
    m << "{\n  \"run_id\": \"" << json_escape(fs::path(dump_dir).filename().string()) << "\",\n"
      << "  \"backend\": \"cpp\",\n"
      << "  \"export\": {\"path\": \"" << json_escape(export_dir) << "\", \"config_sha256\": \""
      << sha256_file(export_dir + "/config.json") << "\"},\n"
      << "  \"tokens\": " << T << ",\n  \"token_range\": [" << a << ", " << b << "],\n"
      << "  \"numerics\": " << numerics_json(opt.num, preset) << ",\n"
      << "  \"bug\": " << (bug_sha.empty() ? "null" : "{\"sha256\": \"" + bug_sha + "\"}") << ",\n"
      << "  \"patches\": [" << patches.str() << "],\n"
      << "  \"patch_skipped_token_ops\": " << tr.patch_skipped_tokens << ",\n"
      << "  \"ops\": [\n    " << ops.str() << "\n  ],\n"
      << "  \"provenance\": {\"timestamp_utc\": \"" << ts << "\", \"git_sha\": \"" << json_escape(git_sha)
      << "\", \"git_dirty\": " << (git_dirty.empty() ? "false" : "true") << ", \"compiler\": \""
      << json_escape(__VERSION__) << "\", \"threads\": "
      << (threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency()))
      << ", \"wall_seconds\": " << secs << "}\n}\n";
    std::cout << "wrote " << written << " ops for " << T << " tokens to " << dump_dir << " in " << secs
              << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "moe_run: error: " << e.what() << "\n";
    return 1;
  }
}
