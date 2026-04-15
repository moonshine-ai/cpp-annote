// SPDX-License-Identifier: MIT
// C++ short-path diarization: community-1 segmentation ORT, then either oracle
// ``hard_clusters`` NPZ or VBx clustering (ORT embedding + PLDA) when embedding paths are set.

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

/// Loads segmentation ONNX, receptive field, and pipeline snapshot once.
  /// When ``embedding_onnx_path`` (and xvec/plda) are non-empty, VBx clustering runs in
  /// ``diarize`` and ``clusters_npz_path`` is ignored. Otherwise oracle NPZ is loaded each call.
class Pyannote {
 public:
  /// Paths default to the CallHome golden bundle used in ``community1_shortpath`` docs
  /// (relative to the process current working directory, as with the CLI).
  explicit Pyannote(
      std::string segmentation_onnx_path = "cpp/artifacts/community1-segmentation.onnx",
      std::string receptive_field_json_path = "cpp/golden/callhome_eng_idx0/receptive_field.json",
      std::string clusters_npz_path =
          "cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s/hard_clusters_final.npz",
      std::string label_mapping_json_path =
          "cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s/label_mapping.json",
      std::string golden_speaker_bounds_json_path =
          "cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s/golden_speaker_bounds.json",
      std::string pipeline_snapshot_json_path = "cpp/golden/callhome_eng_idx0/pipeline_snapshot.json",
      std::string embedding_onnx_path = "",
      std::string xvec_transform_npz_path = "",
      std::string plda_npz_path = "");

  Pyannote(const Pyannote&) = delete;
  Pyannote& operator=(const Pyannote&) = delete;
  Pyannote(Pyannote&&) = delete;
  Pyannote& operator=(Pyannote&&) = delete;

  /// Mono PCM32-ish samples at ``sample_rate`` Hz; resampled internally to the model rate.
  /// ``hard_clusters`` row count must equal the number of segmentation chunks for this audio.
  std::vector<DiarizationTurn> diarize(std::vector<float> audio_data, std::int32_t sample_rate);

  /// Use different oracle artifacts (same ONNX / RF / snapshot). For batch jobs.
  /// If ``golden_speaker_bounds_json_path`` is empty, falls back to the constructor’s bounds path.
  /// In VBx mode ``clusters_npz_path`` may be empty and is not used.
  void set_utterance_paths(
      std::string clusters_npz_path,
      std::string label_mapping_json_path,
      std::string golden_speaker_bounds_json_path = "");

  [[nodiscard]] const std::string& segmentation_onnx_path() const { return onnx_path_; }
  [[nodiscard]] const std::string& clusters_npz_path() const { return clusters_path_; }

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
  std::string clusters_path_;
  std::string label_mapping_path_;
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

  bool vbx_mode_ = false;
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
