// SPDX-License-Identifier: MIT
// C++ diarization engine: community-1 segmentation ORT + embedding ORT + VBx (PLDA).
// Per-chunk building blocks are public for ``StreamingDiarizationSession``.

#pragma once

#include <onnxruntime_cxx_api.h>

#include "clustering_vbx.h"
#include "plda_vbx.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace pyannote {

struct DiarizationProfile {
  int total_chunks = 0;
  int num_frames = 0;
  int num_classes = 0;
  double segmentation_ort_sec = 0.;
  double embedding_ort_sec = 0.;
  double clustering_vbx_sec = 0.;
  double reconstruct_sec = 0.;
  double total_sec = 0.;

  void print(std::ostream& os, const char* prefix = "  ") const {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s%d chunks, %d frames, %d classes\n"
                  "%ssegmentation_ort: %.3fs\n"
                  "%sembedding_ort:    %.3fs\n"
                  "%sclustering_vbx:   %.3fs\n"
                  "%sreconstruct:      %.3fs\n"
                  "%stotal:            %.3fs\n",
                  prefix, total_chunks, num_frames, num_classes,
                  prefix, segmentation_ort_sec,
                  prefix, embedding_ort_sec,
                  prefix, clustering_vbx_sec,
                  prefix, reconstruct_sec,
                  prefix, total_sec);
    os << buf;
  }

  void accumulate(const DiarizationProfile& o) {
    segmentation_ort_sec += o.segmentation_ort_sec;
    embedding_ort_sec += o.embedding_ort_sec;
    clustering_vbx_sec += o.clustering_vbx_sec;
    reconstruct_sec += o.reconstruct_sec;
    total_sec += o.total_sec;
  }
};

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

/// Loads segmentation ONNX and embedding ONNX once. When optional paths are empty, community-1
/// defaults compiled from ``export_cpp_annote_embedded.py`` are used (receptive field, pipeline snapshot,
/// golden speaker bounds, ``xvec_transform`` / ``plda`` tensors). Use via
/// ``StreamingDiarizationSession`` (see ``cpp-annote-streaming.h``).
class CppAnnote {
 public:
  /// ``embedding_onnx_path`` must be non-empty. Leave ``receptive_field_json_path``,
  /// ``golden_speaker_bounds_json_path``, ``pipeline_snapshot_json_path`` empty to use embedded JSON;
  /// leave ``xvec_transform_npz_path`` and ``plda_npz_path`` both empty to use embedded PLDA tensors,
  /// or pass both NPZ paths to override.
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

  // ---- Per-chunk building blocks (used by streaming cache) -----------------

  /// Extract a chunk window from ``audio`` at ``offset``, zero-padded to ``chunk_num_samples``.
  /// Output is ``(num_channels * chunk_num_samples)`` floats.
  static std::vector<float> extract_chunk_audio(
      const float* audio, int64_t num_samples,
      int64_t offset, int chunk_num_samples, int num_channels);

  /// Run segmentation ORT for one prepared chunk.
  /// ``chunk_buf`` has ``(num_channels * chunk_num_samples)`` floats at model rate.
  /// Returns ``(F * K)`` floats.  Sets ``seg_F_`` / ``seg_K_`` on first call.
  std::vector<float> run_segmentation_ort_single(const float* chunk_buf);

  /// Run embedding ORT for all ``K`` local speakers of one chunk.
  /// ``chunk_mono``: ``chunk_num_samples`` floats at model rate.
  /// ``seg_binarized``: ``(F * K)`` from segmentation (multilabel).
  /// Returns ``(K * embed_dim)`` floats.
  std::vector<float> run_embedding_ort_single(
      const float* chunk_mono, const float* seg_binarized);

  /// VBx + reconstruct + binarize from pre-assembled tensors.
  /// ``seg_out``: ``(C * F * K)``, ``emb``: ``(C * K * embed_dim)``.
  /// Fills ``profile.clustering_vbx_sec`` and ``profile.reconstruct_sec``.
  std::vector<DiarizationTurn> cluster_and_decode(
      const std::vector<float>& seg_out,
      const std::vector<float>& emb,
      int C, DiarizationProfile& profile);

  // ---- Accessors -----------------------------------------------------------

  [[nodiscard]] int segmentation_model_sample_rate() const { return cfg_.sr_model; }
  [[nodiscard]] int segmentation_num_channels() const { return cfg_.num_channels; }
  [[nodiscard]] int segmentation_chunk_num_samples() const { return cfg_.chunk_num_samples; }
  [[nodiscard]] double segmentation_chunk_step_sec() const { return cfg_.chunk_step_sec; }
  [[nodiscard]] double segmentation_chunk_duration_sec() const { return cfg_.chunk_dur_sec; }
  [[nodiscard]] int seg_frames_per_chunk() const { return seg_F_; }
  [[nodiscard]] int seg_classes() const { return seg_K_; }
  [[nodiscard]] int embedding_dimension() const { return embed_dim_; }

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
  std::string default_golden_bounds_body_;
  std::string golden_bounds_body_;

  SegConfig cfg_{};
  int seg_F_ = 0;
  int seg_K_ = 0;
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

}  // namespace pyannote
