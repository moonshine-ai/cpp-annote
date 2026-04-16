// SPDX-License-Identifier: MIT
// Realtime-oriented session API: append PCM chunks at arbitrary rates, run full diarization on a
// bounded model-rate buffer on a coarse cadence (VBx / reconstruct are batch over the window).

#pragma once

#include "cpp-annote.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyannote {

struct StreamingDiarizationConfig {
  /// Keep at most this many seconds of audio at the segmentation model sample rate (VBx cost).
  /// If <= 0, buffer grows without trimming (not recommended for long streams).
  double max_buffer_seconds = 120.0;
  /// Run VBx + decode only after this many new analysis chunks since the last refresh (chunking
  /// matches ``CppAnnote::diarize`` sliding windows at model rate).
  int refresh_every_new_chunks = 2;
  /// Minimum wall time between VBx refreshes, in seconds of **ingested media** (``input_end_sec``).
  double refresh_min_interval_sec = 0.5;
};

struct StreamingDiarizationTurn : DiarizationTurn {
  /// Last time this ``(start, end, speaker)`` matched a prior snapshot; bumped when overlap match
  /// fails or bounds/label change beyond tolerance after a refresh.
  double last_updated_at_input_end_sec = 0.;
};

struct StreamingDiarizationSnapshot {
  std::vector<StreamingDiarizationTurn> turns;
  /// Cumulative duration of audio appended on this session (input timeline, from chunk lengths).
  double input_end_sec = 0.;
  /// Time on the input timeline corresponding to ``buffer[0]`` (after trimming).
  double window_start_sec = 0.;
  int refresh_generation = 0;
};

/// Session bound to a ``CppAnnote`` engine; the engine must outlive the session.
class StreamingDiarizationSession {
 public:
  StreamingDiarizationSession(CppAnnote& engine, StreamingDiarizationConfig config = {});

  void start_session();
  /// Append ``num_samples`` mono ``pcm`` at ``sample_rate`` Hz; resamples each chunk to the engine
  /// model rate and concatenates on the session timeline.
  void add_audio_chunk(const float* pcm, std::size_t num_samples, int sample_rate);
  /// Current best snapshot (updated on refresh cadence; ``input_end_sec`` advances every chunk).
  [[nodiscard]] StreamingDiarizationSnapshot snapshot() const;

  /// Final refresh (forces VBx pass if possible) and snapshot.
  StreamingDiarizationSnapshot end_session();

  /// Writes ``StreamingDiarizationSnapshot`` as JSON (metadata + ``turns`` with timestamps).
  static void write_snapshot_json(const std::string& path, const StreamingDiarizationSnapshot& snap);

  StreamingDiarizationSession(const StreamingDiarizationSession&) = delete;
  StreamingDiarizationSession& operator=(const StreamingDiarizationSession&) = delete;

 private:
  void trim_buffer_if_needed();
  void maybe_refresh(bool force);
  static void carry_last_updated_times(
      std::vector<StreamingDiarizationTurn>& next,
      const std::vector<StreamingDiarizationTurn>& prev,
      double input_end_sec);

  struct CachedChunk {
    std::vector<float> seg;  // (F * K)
    std::vector<float> emb;  // (K * dim)
  };

  CppAnnote& engine_;
  StreamingDiarizationConfig cfg_{};

  std::vector<float> buffer_;
  double input_end_sec_ = 0.;
  double window_start_sec_ = 0.;
  int64_t buffer_abs_start_samples_ = 0;

  std::unordered_map<int64_t, CachedChunk> chunk_cache_;

  int last_refresh_total_chunks_ = -1;
  std::optional<double> last_refresh_at_input_end_;

  DiarizationProfile cumulative_profile_{};
  int refresh_count_ = 0;

  StreamingDiarizationSnapshot snapshot_;
};

}  // namespace pyannote
