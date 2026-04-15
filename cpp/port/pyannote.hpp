// SPDX-License-Identifier: MIT
// C++ short-path diarization: community-1 segmentation ORT, then VBx clustering
// (ORT embedding + PLDA) in ``diarize``.

#pragma once

#include <onnxruntime_cxx_api.h>

#include "clustering_vbx.hpp"
#include "plda_vbx.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pyannote {

struct DiarizationTurn {
  double start = 0.;
  double end = 0.;
  std::string speaker;
  bool operator<(const DiarizationTurn& o) const {
    if (start != o.start) {
      return start < o.start;
    }
    if (end != o.end) {
      return end < o.end;
    }
    return speaker < o.speaker;
  }
};

/// Loads segmentation ONNX, embedding ONNX, PLDA, receptive field, and pipeline snapshot once.
/// Cluster assignments are produced in C++ via VBx in ``diarize``.
class CppAnnote {
 public:
  /// ``embedding_onnx_path``, ``xvec_transform_npz_path``, and ``plda_npz_path`` must be non-empty.
  /// Other paths default to the CallHome golden bundle layout (relative to process cwd).
  explicit CppAnnote(
      std::string segmentation_onnx_path,
      std::string receptive_field_json_path,
      std::string golden_speaker_bounds_json_path,
      std::string pipeline_snapshot_json_path,
      std::string embedding_onnx_path,
      std::string xvec_transform_npz_path,
      std::string plda_npz_path);

  CppAnnote(const CppAnnote&) = delete;
  CppAnnote& operator=(const CppAnnote&) = delete;
  CppAnnote(CppAnnote&&) = delete;
  CppAnnote& operator=(CppAnnote&&) = delete;

  /// Mono PCM32-ish samples at ``sample_rate`` Hz; resampled internally to the model rate.
  std::vector<DiarizationTurn> diarize(std::vector<float> audio_data, std::int32_t sample_rate);

  /// Optional per-utterance ``golden_speaker_bounds.json`` for batch jobs. Empty string keeps the
  /// constructor default.
  void set_golden_speaker_bounds(std::string golden_speaker_bounds_json_path);

  [[nodiscard]] const std::string& segmentation_onnx_path() const { return onnx_path_; }

 private:
  struct SegConfig {
    int sr_model = 0;
    int num_channels = 0;
    int chunk_num_samples = 0;
    bool multilabel_export = false;
    double chunk_step_sec = 0.;
    double chunk_dur_sec = 0.;
  };

  std::string onnx_path_;
  std::string receptive_field_path_;
  std::string golden_bounds_path_;
  std::string default_golden_bounds_path_;
  std::string pipeline_snapshot_path_;

  SegConfig cfg_{};
  double rf_dur_ = 0.;
  double rf_step_ = 0.;
  double min_off_ = 0.;
  double min_on_ = 0.;

  Ort::Env ort_env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::MemoryInfo mem_;
  Ort::AllocatorWithDefaultOptions alloc_;
  Ort::AllocatedStringPtr in_name_;
  Ort::AllocatedStringPtr out_name_;

  std::string embedding_onnx_path_;
  std::string xvec_npz_path_;
  std::string plda_npz_path_;
  int embed_sr_ = 16000;
  int embed_mel_bins_ = 80;
  float embed_frame_length_ms_ = 25.f;
  float embed_frame_shift_ms_ = 10.f;
  int embed_dim_ = 256;
  bool embedding_exclude_overlap_ = false;
  int min_num_samples_ = 800;

  std::unique_ptr<Ort::Session> embed_session_;
  /// If true, ORT input 0 is ``fbank`` and input 1 is ``weights`` (names from sidecar JSON).
  bool embed_inputs_fbank_then_weights_ = true;

  std::unique_ptr<plda_vbx::PldaModel> plda_model_;
  clustering_vbx::VbxClusteringParams vbx_params_{};
};

void write_diarization_json(const std::string& path, const std::vector<DiarizationTurn>& turns);

}  // namespace pyannote
