#include "io.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace coinflip {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// ---------------------------------------------------------------- .npy
size_t NpyArray::size() const {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return n;
}

namespace {

std::string header_field(const std::string& h, const std::string& key) {
  const auto k = h.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy header lacks " + key);
  const auto colon = h.find(':', k);
  auto start = h.find_first_not_of(" ", colon + 1);
  if (h[start] == '\'') return h.substr(start + 1, h.find('\'', start + 1) - start - 1);
  if (h[start] == '(') return h.substr(start + 1, h.find(')', start) - start - 1);
  const auto end = h.find_first_of(",}", start);
  return h.substr(start, end - start);
}

void write_header(std::ofstream& f, const std::string& descr, const std::vector<size_t>& shape) {
  std::ostringstream d;
  d << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) d << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  d << "), }";
  std::string h = d.str();
  const size_t total = 10 + h.size() + 1;
  h.append((64 - total % 64) % 64, ' ');
  h.push_back('\n');
  const uint16_t len = static_cast<uint16_t>(h.size());
  f.write("\x93NUMPY\x01\x00", 8);
  f.write(reinterpret_cast<const char*>(&len), 2);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
}

}  // namespace

NpyArray npy_read(const std::string& path) {
  const std::string raw = read_file(path);
  if (raw.size() < 10 || raw.compare(0, 6, "\x93NUMPY") != 0) throw std::runtime_error(path + ": not an .npy file");
  const int major = static_cast<unsigned char>(raw[6]);
  size_t hlen, off;
  if (major == 1) {
    hlen = static_cast<unsigned char>(raw[8]) | (static_cast<unsigned char>(raw[9]) << 8);
    off = 10;
  } else {
    hlen = 0;
    for (int i = 0; i < 4; ++i) hlen |= static_cast<size_t>(static_cast<unsigned char>(raw[8 + i])) << (8 * i);
    off = 12;
  }
  const std::string h = raw.substr(off, hlen);
  NpyArray a;
  a.descr = header_field(h, "descr");
  if (header_field(h, "fortran_order").find("False") == std::string::npos)
    throw std::runtime_error(path + ": fortran order not supported");
  std::stringstream sh(header_field(h, "shape"));
  for (std::string tok; std::getline(sh, tok, ',');) {
    tok.erase(0, tok.find_first_not_of(' '));
    if (!tok.empty()) a.shape.push_back(std::stoull(tok));
  }
  const size_t n = a.size();
  const char* p = raw.data() + off + hlen;
  const size_t avail = raw.size() - off - hlen;
  a.data.resize(n);
  auto need = [&](size_t bytes) {
    if (avail < n * bytes) throw std::runtime_error(path + ": truncated data");
  };
  if (a.descr == "<f4") {
    need(4);
    for (size_t i = 0; i < n; ++i) { float v; std::memcpy(&v, p + 4 * i, 4); a.data[i] = v; }
  } else if (a.descr == "<f8") {
    need(8);
    for (size_t i = 0; i < n; ++i) { double v; std::memcpy(&v, p + 8 * i, 8); a.data[i] = v; }
  } else if (a.descr == "<i4") {
    need(4);
    for (size_t i = 0; i < n; ++i) { int32_t v; std::memcpy(&v, p + 4 * i, 4); a.data[i] = v; }
  } else if (a.descr == "<i8") {
    need(8);
    for (size_t i = 0; i < n; ++i) { int64_t v; std::memcpy(&v, p + 8 * i, 8); a.data[i] = static_cast<double>(v); }
  } else {
    throw std::runtime_error(path + ": unsupported dtype " + a.descr);
  }
  return a;
}

std::vector<float> npy_read_f32(const std::string& path, const std::vector<size_t>& expect) {
  NpyArray a = npy_read(path);
  if (a.descr != "<f4") throw std::runtime_error(path + ": expected <f4, got " + a.descr);
  if (!expect.empty() && a.shape != expect) {
    std::ostringstream m;
    m << path << ": shape mismatch (got";
    for (auto d : a.shape) m << ' ' << d;
    m << ", expected";
    for (auto d : expect) m << ' ' << d;
    m << ")";
    throw std::runtime_error(m.str());
  }
  return std::vector<float>(a.data.begin(), a.data.end());
}

void npy_write_f32(const std::string& path, const std::vector<size_t>& shape, const double* v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  write_header(f, "<f4", shape);
  size_t n = 1;
  for (size_t d : shape) n *= d;
  std::vector<float> buf(n);
  for (size_t i = 0; i < n; ++i) buf[i] = static_cast<float>(v[i]);
  f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(n * 4));
}

void npy_write_i32(const std::string& path, const std::vector<size_t>& shape, const int* v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  write_header(f, "<i4", shape);
  size_t n = 1;
  for (size_t d : shape) n *= d;
  f.write(reinterpret_cast<const char*>(v), static_cast<std::streamsize>(n * 4));
}

// ---------------------------------------------------------------- JSON
const Json& Json::at(const std::string& k) const {
  auto it = obj.find(k);
  if (kind != Kind::Object || it == obj.end()) throw std::runtime_error("json: missing key '" + k + "'");
  return it->second;
}

std::string Json::as_string() const {
  switch (kind) {
    case Kind::String: return str;
    case Kind::Bool: return b ? "true" : "false";
    case Kind::Number: {
      if (num == std::floor(num) && std::fabs(num) < 1e15) return std::to_string(static_cast<long long>(num));
      std::ostringstream s;
      s << std::setprecision(17) << num;
      return s.str();
    }
    case Kind::Null: return "null";
    default: throw std::runtime_error("json: value is not a scalar");
  }
}

namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;
  void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
  [[noreturn]] void fail(const std::string& m) { throw std::runtime_error("json: " + m + " at offset " + std::to_string(i)); }
  bool lit(const char* w) {
    const size_t n = std::strlen(w);
    if (s.compare(i, n, w) == 0) { i += n; return true; }
    return false;
  }
  std::string string() {
    if (s[i] != '"') fail("expected string");
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
      char c = s[i++];
      if (c == '\\') {
        if (i >= s.size()) fail("bad escape");
        char e = s[i++];
        switch (e) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            const unsigned cp = static_cast<unsigned>(std::stoul(s.substr(i, 4), nullptr, 16));
            i += 4;
            if (cp < 0x80) out += static_cast<char>(cp);
            else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
            else { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
            break;
          }
          default: out += e;
        }
      } else {
        out += c;
      }
    }
    if (i >= s.size()) fail("unterminated string");
    ++i;
    return out;
  }
  Json value() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    Json j;
    const char c = s[i];
    if (c == '{') {
      j.kind = Json::Kind::Object;
      ++i;
      ws();
      if (s[i] == '}') { ++i; return j; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (s[i++] != ':') fail("expected ':'");
        j.obj[k] = value();
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == '}') { ++i; return j; }
        fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      j.kind = Json::Kind::Array;
      ++i;
      ws();
      if (s[i] == ']') { ++i; return j; }
      while (true) {
        j.arr.push_back(value());
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == ']') { ++i; return j; }
        fail("expected ',' or ']'");
      }
    }
    if (c == '"') { j.kind = Json::Kind::String; j.str = string(); return j; }
    if (lit("true")) { j.kind = Json::Kind::Bool; j.b = true; return j; }
    if (lit("false")) { j.kind = Json::Kind::Bool; j.b = false; return j; }
    if (lit("null")) return j;
    size_t end = i;
    while (end < s.size() && std::strchr("+-0123456789.eE", s[end])) ++end;
    if (end == i) fail("unexpected character");
    j.kind = Json::Kind::Number;
    j.num = std::stod(s.substr(i, end - i));
    i = end;
    return j;
  }
};

}  // namespace

Json json_parse(const std::string& text) {
  Parser p{text};
  Json j = p.value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return j;
}

Json json_read_file(const std::string& path) { return json_parse(read_file(path)); }

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)
std::string sha256_hex(const std::string& msg) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
      0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
      0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::string m = msg;
  const uint64_t bits = static_cast<uint64_t>(msg.size()) * 8;
  m.push_back(static_cast<char>(0x80));
  while (m.size() % 64 != 56) m.push_back('\0');
  for (int i = 7; i >= 0; --i) m.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (size_t c = 0; c < m.size(); c += 64) {
    uint32_t w[64];
    for (int t = 0; t < 16; ++t)
      w[t] = (static_cast<uint32_t>(static_cast<unsigned char>(m[c + 4 * t])) << 24) |
             (static_cast<uint32_t>(static_cast<unsigned char>(m[c + 4 * t + 1])) << 16) |
             (static_cast<uint32_t>(static_cast<unsigned char>(m[c + 4 * t + 2])) << 8) |
             static_cast<uint32_t>(static_cast<unsigned char>(m[c + 4 * t + 3]));
    for (int t = 16; t < 64; ++t) {
      const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], cc = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int t = 0; t < 64; ++t) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + k[t] + w[t];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
      const uint32_t t2 = S0 + mj;
      hh = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += cc; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  std::ostringstream o;
  for (uint32_t v : h) o << std::hex << std::setw(8) << std::setfill('0') << v;
  return o.str();
}

std::string sha256_file(const std::string& path) { return sha256_hex(read_file(path)); }

}  // namespace coinflip
