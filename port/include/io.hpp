// Minimal I/O for the port: NumPy .npy arrays, a small JSON reader, and SHA-256 (no third-party dependencies).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace coinflip {

// ---- .npy ----
struct NpyArray {
  std::string descr;           // "<f4", "<i4", "<f8", "<i8"
  std::vector<size_t> shape;
  std::vector<double> data;    // values widened to double (C order)
  size_t size() const;
};
NpyArray npy_read(const std::string& path);
std::vector<float> npy_read_f32(const std::string& path, const std::vector<size_t>& expect_shape);
void npy_write_f32(const std::string& path, const std::vector<size_t>& shape, const double* values);
void npy_write_i32(const std::string& path, const std::vector<size_t>& shape, const int* values);

// ---- JSON (read side) ----
struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
  bool b = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool has(const std::string& k) const { return kind == Kind::Object && obj.count(k) > 0; }
  const Json& at(const std::string& k) const;
  std::string as_string() const;  // numbers and bools are converted to their text form
};
Json json_parse(const std::string& text);
Json json_read_file(const std::string& path);
std::string json_escape(const std::string& s);

// ---- misc ----
std::string read_file(const std::string& path);
std::string sha256_hex(const std::string& bytes);
std::string sha256_file(const std::string& path);

}  // namespace coinflip
