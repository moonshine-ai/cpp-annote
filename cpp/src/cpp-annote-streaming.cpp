// SPDX-License-Identifier: MIT

#include "cpp-annote-streaming.h"

#include "wav_pcm_float32.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace pyannote {
namespace {

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += c;
  }
  return o;
}

double segment_iou(double a0, double a1, double b0, double b1) {
  const double inter = std::max(0., std::min(a1, b1) - std::max(a0, b0));
  const double span = std::max(a1, b1) - std::min(a0, b0);
  if (span <= 1e-12) {
    return 0.;
  }
  return inter / span;
}


}  // namespace

StreamingDiarizationSession::StreamingDiarizationSession(CppAnnote& engine, StreamingDiarizationConfig config)
    : engine_(engine), cfg_(std::move(config)) {
  cfg_.refresh_every_new_chunks = std::max(1, cfg_.refresh_every_new_chunks);
}

void StreamingDiarizationSession::start_session() {
  buffer_.clear();
  input_end_sec_ = 0.;
  window_start_sec_ = 0.;
  buffer_abs_start_samples_ = 0;
  chunk_cache_.clear();
  last_refresh_total_chunks_ = -1;
  last_refresh_at_input_end_.reset();
  cumulative_profile_ = DiarizationProfile{};
  refresh_count_ = 0;
  snapshot_ = StreamingDiarizationSnapshot{};
}

void StreamingDiarizationSession::trim_buffer_if_needed() {
  if (cfg_.max_buffer_seconds <= 0.) {
    return;
  }
  const int sr = engine_.segmentation_model_sample_rate();
  if (sr <= 0) {
    return;
  }
  const std::size_t cap =
      static_cast<std::size_t>(std::max(1., cfg_.max_buffer_seconds) * static_cast<double>(sr));
  if (buffer_.size() <= cap) {
    return;
  }
  const std::size_t drop = buffer_.size() - cap;
  buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(drop));
  window_start_sec_ += static_cast<double>(drop) / static_cast<double>(sr);
  buffer_abs_start_samples_ += static_cast<int64_t>(drop);

  // Evict cache entries that fell off the front.
  for (auto it = chunk_cache_.begin(); it != chunk_cache_.end();) {
    if (it->first < buffer_abs_start_samples_) {
      it = chunk_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void StreamingDiarizationSession::add_audio_chunk(const float* pcm, std::size_t num_samples, int sample_rate) {
  if (pcm == nullptr || num_samples == 0) {
    snapshot_.input_end_sec = input_end_sec_;
    return;
  }
  if (sample_rate <= 0) {
    throw std::runtime_error("StreamingDiarizationSession: sample_rate must be positive");
  }
  const int sr_model = engine_.segmentation_model_sample_rate();
  std::vector<float> chunk(pcm, pcm + num_samples);
  std::vector<float> res =
      wav_pcm::linear_resample(chunk, sample_rate, sr_model);
  buffer_.insert(buffer_.end(), res.begin(), res.end());
  input_end_sec_ += static_cast<double>(num_samples) / static_cast<double>(sample_rate);
  trim_buffer_if_needed();
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  maybe_refresh(false);
}

void StreamingDiarizationSession::carry_last_updated_times(
    std::vector<StreamingDiarizationTurn>& next,
    const std::vector<StreamingDiarizationTurn>& prev,
    double input_end_sec) {
  constexpr double kTol = 0.25;
  constexpr double kIouMin = 0.2;
  for (auto& t : next) {
    t.last_updated_at_input_end_sec = input_end_sec;
    double best_iou = 0.;
    const StreamingDiarizationTurn* best = nullptr;
    for (const auto& p : prev) {
      const double i = segment_iou(t.start, t.end, p.start, p.end);
      if (i > best_iou) {
        best_iou = i;
        best = &p;
      }
    }
    if (best != nullptr && best_iou >= kIouMin && best->speaker == t.speaker &&
        std::abs(t.start - best->start) < kTol && std::abs(t.end - best->end) < kTol) {
      t.last_updated_at_input_end_sec = best->last_updated_at_input_end_sec;
    }
  }
}

void StreamingDiarizationSession::maybe_refresh(bool force) {
  using Clock = std::chrono::steady_clock;

  const int sr_model = engine_.segmentation_model_sample_rate();
  const int num_channels = engine_.segmentation_num_channels();
  const int chunk_num_samples = engine_.segmentation_chunk_num_samples();
  const double chunk_step_sec = engine_.segmentation_chunk_step_sec();
  const int step_samples = static_cast<int>(std::lrint(chunk_step_sec * static_cast<double>(sr_model)));
  if (step_samples <= 0 || chunk_num_samples <= 0) {
    return;
  }

  const int64_t num_samples_i = static_cast<int64_t>(buffer_.size());
  int64_t num_complete_chunks = 0;
  if (num_samples_i >= chunk_num_samples) {
    num_complete_chunks = (num_samples_i - chunk_num_samples) / step_samples + 1;
  }
  const bool has_last =
      (num_samples_i < chunk_num_samples) || ((num_samples_i - chunk_num_samples) % step_samples > 0);
  const int64_t total_chunks = num_complete_chunks + (has_last ? 1 : 0);
  if (total_chunks <= 0) {
    return;
  }

  if (!force) {
    if (last_refresh_total_chunks_ >= 0) {
      if (total_chunks < last_refresh_total_chunks_ + cfg_.refresh_every_new_chunks) {
        return;
      }
    }
    if (last_refresh_at_input_end_.has_value()) {
      if (input_end_sec_ - *last_refresh_at_input_end_ < cfg_.refresh_min_interval_sec) {
        return;
      }
    }
  }

  const int C = static_cast<int>(total_chunks);
  int new_seg_count = 0;
  int new_emb_count = 0;

  const auto t_seg_start = Clock::now();

  // Segmentation pass — use cache for complete chunks, always re-run the partial tail.
  for (int64_t c = 0; c < total_chunks; ++c) {
    const int64_t buf_off = (c < num_complete_chunks) ? c * step_samples : num_complete_chunks * step_samples;
    const int64_t abs_off = buffer_abs_start_samples_ + buf_off;
    const bool is_tail = (c == total_chunks - 1 && has_last);
    if (!is_tail && chunk_cache_.count(abs_off)) {
      continue;
    }
    auto chunk_buf = CppAnnote::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples, num_channels);
    auto seg = engine_.run_segmentation_ort_single(chunk_buf.data());
    auto mono = CppAnnote::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples, 1);
    auto emb_chunk = engine_.run_embedding_ort_single(mono.data(), seg.data());
    chunk_cache_[abs_off] = CachedChunk{std::move(seg), std::move(emb_chunk)};
    ++new_seg_count;
    ++new_emb_count;
  }

  const auto t_after_seg_emb = Clock::now();

  // Assemble full tensors from cache.
  const int F = engine_.seg_frames_per_chunk();
  const int K = engine_.seg_classes();
  const int dim = engine_.embedding_dimension();
  const int FK = F * K;

  std::vector<float> seg_out(static_cast<size_t>(C) * static_cast<size_t>(FK));
  std::vector<float> emb_all(
      static_cast<size_t>(C) * static_cast<size_t>(K) * static_cast<size_t>(dim),
      std::numeric_limits<float>::quiet_NaN());

  int64_t tail_abs_off = -1;
  for (int64_t c = 0; c < total_chunks; ++c) {
    const int64_t buf_off = (c < num_complete_chunks) ? c * step_samples : num_complete_chunks * step_samples;
    const int64_t abs_off = buffer_abs_start_samples_ + buf_off;
    const bool is_tail = (c == total_chunks - 1 && has_last);
    if (is_tail) tail_abs_off = abs_off;
    const auto& cached = chunk_cache_.at(abs_off);
    std::memcpy(&seg_out[static_cast<size_t>(c) * static_cast<size_t>(FK)],
                cached.seg.data(), static_cast<size_t>(FK) * sizeof(float));
    std::memcpy(&emb_all[static_cast<size_t>(c) * static_cast<size_t>(K) * static_cast<size_t>(dim)],
                cached.emb.data(), static_cast<size_t>(K) * static_cast<size_t>(dim) * sizeof(float));
  }

  // Evict the tail chunk from cache — it was computed with zero-padded audio
  // and its content will change as more audio arrives.
  if (tail_abs_off >= 0) {
    chunk_cache_.erase(tail_abs_off);
  }

  DiarizationProfile prof;
  prof.segmentation_ort_sec = 0.;
  prof.embedding_ort_sec = std::chrono::duration<double>(t_after_seg_emb - t_seg_start).count();

  std::vector<DiarizationTurn> raw = engine_.cluster_and_decode(seg_out, emb_all, C, prof);

  prof.segmentation_ort_sec = 0.;
  prof.total_sec = std::chrono::duration<double>(Clock::now() - t_seg_start).count();
  cumulative_profile_.accumulate(prof);
  ++refresh_count_;

  char logbuf[384];
  std::snprintf(logbuf, sizeof(logbuf),
                "[streaming refresh #%d] chunks=%d (new_ort=%d)  ort=%.3fs  vbx=%.3fs  recon=%.3fs  total=%.3fs  buf=%.1fs  cache=%zu",
                refresh_count_, C, new_seg_count,
                prof.embedding_ort_sec,
                prof.clustering_vbx_sec, prof.reconstruct_sec,
                prof.total_sec,
                static_cast<double>(buffer_.size()) / static_cast<double>(std::max(1, sr_model)),
                chunk_cache_.size());
  std::cerr << logbuf << "\n";
  std::vector<StreamingDiarizationTurn> next;
  next.reserve(raw.size());
  const double origin = window_start_sec_;
  for (const DiarizationTurn& t : raw) {
    StreamingDiarizationTurn st;
    static_cast<DiarizationTurn&>(st) = t;
    st.start += origin;
    st.end += origin;
    next.push_back(st);
  }

  const std::vector<StreamingDiarizationTurn> prev = std::move(snapshot_.turns);
  carry_last_updated_times(next, prev, input_end_sec_);

  snapshot_.turns = std::move(next);
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  ++snapshot_.refresh_generation;

  last_refresh_total_chunks_ = static_cast<int>(total_chunks);
  last_refresh_at_input_end_ = input_end_sec_;
}

StreamingDiarizationSnapshot StreamingDiarizationSession::snapshot() const {
  return snapshot_;
}

StreamingDiarizationSnapshot StreamingDiarizationSession::end_session() {
  maybe_refresh(true);
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;

  std::cerr << "[streaming summary] " << refresh_count_ << " refreshes, cumulative breakdown:\n";
  cumulative_profile_.print(std::cerr, "  ");

  return snapshot_;
}

void StreamingDiarizationSession::write_snapshot_json(const std::string& path, const StreamingDiarizationSnapshot& snap) {
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("write_streaming_snapshot_json: open failed: " + path);
  }
  f << std::setprecision(17);
  f << "{\n";
  f << "  \"input_end_sec\": " << snap.input_end_sec << ",\n";
  f << "  \"window_start_sec\": " << snap.window_start_sec << ",\n";
  f << "  \"refresh_generation\": " << snap.refresh_generation << ",\n";
  f << "  \"turns\": [\n";
  for (size_t i = 0; i < snap.turns.size(); ++i) {
    const StreamingDiarizationTurn& t = snap.turns[i];
    f << "    {\"start\": " << t.start << ", \"end\": " << t.end << ", \"speaker\": \"" << json_escape(t.speaker)
      << "\", \"last_updated_at_input_end_sec\": " << t.last_updated_at_input_end_sec << "}";
    if (i + 1 < snap.turns.size()) {
      f << ",";
    }
    f << "\n";
  }
  f << "  ]\n}\n";
}

}  // namespace pyannote
