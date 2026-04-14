// SPDX-License-Identifier: MIT
// Compare ONNX Runtime embedding output to Python golden (fbank + weights -> embedding).

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnpy.h"

static std::string read_text_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("failed to open: " + path);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static int json_int_field(const std::string& json, const char* key) {
  std::string pat = std::string("\"") + key + "\"\\s*:\\s*([0-9]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("JSON parse: missing int field \"") + key + "\"");
  }
  return std::stoi(m[1].str());
}

static float max_abs_diff(const float* a, const float* b, size_t n) {
  float m = 0.f;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: embedding_golden_test <community1-embedding.onnx> <golden_utterance_dir>\n";
    std::cerr << "  requires embedding_chunk0_spk0_ort.npz (re-run dump_diarization_golden.py)\n";
    return 2;
  }
  const std::string onnx_path = argv[1];
  const std::string golden_dir = argv[2];
  if (onnx_path.size() < 5 || onnx_path.substr(onnx_path.size() - 5) != ".onnx") {
    std::cerr << "Expected .onnx path\n";
    return 2;
  }
  const std::string json_path = onnx_path.substr(0, onnx_path.size() - 5) + ".json";
  const std::string npz_path = golden_dir + "/embedding_chunk0_spk0_ort.npz";

  const std::string json = read_text_file(json_path);
  const int embedding_dim = json_int_field(json, "embedding_dim");

  cnpy::npz_t npz = cnpy::npz_load(npz_path);
  if (!npz.count("fbank") || !npz.count("weights") || !npz.count("expected_embedding")) {
    throw std::runtime_error(
        "missing keys in embedding_chunk0_spk0_ort.npz — re-run cpp/scripts/dump_diarization_golden.py");
  }
  const cnpy::NpyArray& fb = npz["fbank"];
  const cnpy::NpyArray& wt = npz["weights"];
  const cnpy::NpyArray& exp = npz["expected_embedding"];

  if (fb.shape.size() != 3 || wt.shape.size() != 2 || exp.shape.size() != 1) {
    throw std::runtime_error("unexpected tensor ranks in golden npz");
  }
  if (static_cast<int>(exp.shape[0]) != embedding_dim) {
    throw std::runtime_error("expected_embedding dim does not match embedding.json");
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embedding_golden_test");
  Ort::SessionOptions session_options;
  Ort::Session session(env, onnx_path.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions alloc;

  Ort::AllocatedStringPtr in0 = session.GetInputNameAllocated(0, alloc);
  Ort::AllocatedStringPtr in1 = session.GetInputNameAllocated(1, alloc);
  Ort::AllocatedStringPtr out0 = session.GetOutputNameAllocated(0, alloc);

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const float* fbank_ptr = fb.data<float>();
  const float* weights_ptr = wt.data<float>();
  std::array<int64_t, 3> fbank_shape{
      static_cast<int64_t>(fb.shape[0]),
      static_cast<int64_t>(fb.shape[1]),
      static_cast<int64_t>(fb.shape[2]),
  };
  std::array<int64_t, 2> weights_shape{
      static_cast<int64_t>(wt.shape[0]),
      static_cast<int64_t>(wt.shape[1]),
  };

  Ort::Value fbank_tensor = Ort::Value::CreateTensor<float>(
      mem,
      const_cast<float*>(fbank_ptr),
      fb.num_vals,
      fbank_shape.data(),
      fbank_shape.size());
  Ort::Value weights_tensor = Ort::Value::CreateTensor<float>(
      mem,
      const_cast<float*>(weights_ptr),
      wt.num_vals,
      weights_shape.data(),
      weights_shape.size());

  const char* in_names[] = {in0.get(), in1.get()};
  Ort::Value inputs[2];
  if (std::string(in0.get()) == "fbank") {
    inputs[0] = std::move(fbank_tensor);
    inputs[1] = std::move(weights_tensor);
  } else {
    inputs[0] = std::move(weights_tensor);
    inputs[1] = std::move(fbank_tensor);
  }
  const char* out_names[] = {out0.get()};
  auto outs = session.Run(
      Ort::RunOptions{nullptr},
      in_names,
      inputs,
      2,
      out_names,
      1);

  float* out_ptr = outs[0].GetTensorMutableData<float>();
  auto out_info = outs[0].GetTensorTypeAndShapeInfo();
  const auto out_shape = out_info.GetShape();
  size_t out_n = 1;
  for (int64_t d : out_shape) {
    out_n *= static_cast<size_t>(d);
  }
  if (out_n != static_cast<size_t>(embedding_dim)) {
    std::ostringstream oss;
    oss << "ORT output size " << out_n << " != embedding_dim " << embedding_dim;
    throw std::runtime_error(oss.str());
  }

  const float* exp_ptr = exp.data<float>();
  const float mad = max_abs_diff(out_ptr, exp_ptr, embedding_dim);
  const float rtol = 1e-2f;
  const float atol = 1e-3f;
  bool pass = true;
  for (int i = 0; i < embedding_dim; ++i) {
    const float a = out_ptr[i];
    const float b = exp_ptr[i];
    if (!(std::abs(a - b) <= atol + rtol * std::abs(b))) {
      pass = false;
      break;
    }
  }

  std::cout << "max_abs_diff(embedding) = " << mad << "\n";
  if (!pass) {
    std::cerr << "FAIL: embedding allclose(rtol=" << rtol << ", atol=" << atol << ")\n";
    return 1;
  }
  std::cout << "PASS embedding chunk0 spk0 (ONNX Runtime vs golden NPZ)\n";
  return 0;
}
