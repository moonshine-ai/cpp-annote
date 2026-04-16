// SPDX-License-Identifier: MIT
// Comparison tool: runs both batch and streaming paths on a single WAV file and
// reports per-stage numeric differences to identify accuracy divergence sources.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpp-annote.h"
#include "cpp-annote-streaming.h"
#include "wav_pcm_float32.h"

namespace fs = std::filesystem;

static std::string get_arg(int argc, char** argv, const char* key, const std::string& def = "") {
  for (int i = 1; i < argc - 1; ++i) {
    if (std::string(argv[i]) == key) {
      return argv[i + 1];
    }
  }
  return def;
}

static bool has_flag(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == key) {
      return true;
    }
  }
  return false;
}

struct VecStats {
  size_t total = 0;
  size_t nan_a = 0;
  size_t nan_b = 0;
  double max_abs_diff = 0.;
  double mean_abs_diff = 0.;
  double rms_diff = 0.;
};

static VecStats compare_vectors(const std::vector<float>& a, const std::vector<float>& b,
                                const char* label) {
  VecStats s;
  s.total = std::max(a.size(), b.size());
  if (a.size() != b.size()) {
    std::fprintf(stderr, "  [%s] SIZE MISMATCH: %zu vs %zu\n", label, a.size(), b.size());
  }
  const size_t n = std::min(a.size(), b.size());
  double sum_abs = 0., sum_sq = 0.;
  size_t valid = 0;
  for (size_t i = 0; i < n; ++i) {
    const bool a_nan = std::isnan(a[i]);
    const bool b_nan = std::isnan(b[i]);
    if (a_nan) ++s.nan_a;
    if (b_nan) ++s.nan_b;
    if (a_nan || b_nan) continue;
    const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    s.max_abs_diff = std::max(s.max_abs_diff, d);
    sum_abs += d;
    sum_sq += d * d;
    ++valid;
  }
  if (valid > 0) {
    s.mean_abs_diff = sum_abs / static_cast<double>(valid);
    s.rms_diff = std::sqrt(sum_sq / static_cast<double>(valid));
  }
  std::fprintf(stderr,
               "  [%s] n=%zu  max_diff=%.6e  mean_diff=%.6e  rms=%.6e  nan_a=%zu nan_b=%zu\n",
               label, n, s.max_abs_diff, s.mean_abs_diff, s.rms_diff, s.nan_a, s.nan_b);
  return s;
}

static void compare_turns(const std::vector<pyannote::DiarizationTurn>& a,
                          const std::vector<pyannote::DiarizationTurn>& b) {
  std::fprintf(stderr, "  [turns] batch=%zu  streaming=%zu\n", a.size(), b.size());
  if (a.size() != b.size()) {
    std::fprintf(stderr, "  [turns] COUNT DIFFERS\n");
  }
  const size_t n = std::min(a.size(), b.size());
  int label_mismatches = 0;
  double max_start_diff = 0., max_end_diff = 0.;
  for (size_t i = 0; i < n; ++i) {
    max_start_diff = std::max(max_start_diff, std::abs(a[i].start - b[i].start));
    max_end_diff = std::max(max_end_diff, std::abs(a[i].end - b[i].end));
    if (a[i].speaker != b[i].speaker) ++label_mismatches;
  }
  std::fprintf(stderr,
               "  [turns] max_start_diff=%.6f  max_end_diff=%.6f  label_mismatches=%d/%zu\n",
               max_start_diff, max_end_diff, label_mismatches, n);
  if (n <= 40 && a.size() != b.size()) {
    std::fprintf(stderr, "  -- batch turns:\n");
    for (const auto& t : a) {
      std::fprintf(stderr, "     [%.3f, %.3f] %s\n", t.start, t.end, t.speaker.c_str());
    }
    std::fprintf(stderr, "  -- streaming turns:\n");
    for (const auto& t : b) {
      std::fprintf(stderr, "     [%.3f, %.3f] %s\n", t.start, t.end, t.speaker.c_str());
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help")) {
    std::cerr
        << "streaming_vs_batch_test — Run both batch and streaming on one WAV, compare intermediates.\n\n"
        << "Required:\n"
        << "  --wav PATH\n"
        << "  --segmentation-onnx PATH\n"
        << "  --embedding-onnx PATH\n\n"
        << "Optional:\n"
        << "  --golden-speaker-bounds PATH\n"
        << "  --refresh-every N            re-cluster every N seconds of new audio (default 999999 = once at end)\n";
    return 2;
  }

  try {
    const std::string seg_onnx = get_arg(argc, argv, "--segmentation-onnx");
    const std::string emb_onnx = get_arg(argc, argv, "--embedding-onnx");
    const std::string wav_path = get_arg(argc, argv, "--wav");
    const std::string bounds_path = get_arg(argc, argv, "--golden-speaker-bounds");
    const std::string ref_str = get_arg(argc, argv, "--refresh-every");
    const double refresh_every_sec = ref_str.empty() ? 999999.0 : std::stod(ref_str);

    if (seg_onnx.empty() || emb_onnx.empty() || wav_path.empty()) {
      throw std::runtime_error("--wav, --segmentation-onnx, and --embedding-onnx are required");
    }

    pyannote::CppAnnote engine(seg_onnx, "", bounds_path, "", emb_onnx, "", "");
    const int sr_model = engine.segmentation_model_sample_rate();

    int wav_sr = 0;
    std::vector<float> raw_audio = wav_pcm::load_wav_pcm16_mono_float32(wav_path, wav_sr);
    const double audio_sec = wav_sr > 0
        ? static_cast<double>(raw_audio.size()) / static_cast<double>(wav_sr) : 0.;
    std::fprintf(stderr, "=== WAV: %s  (%.2fs, sr=%d, %zu samples) ===\n",
                 wav_path.c_str(), audio_sec, wav_sr, raw_audio.size());

    // ---- Stage 0: Compare resampled audio ----
    std::fprintf(stderr, "\n--- Stage 0: Resampled audio comparison ---\n");

    std::vector<float> batch_resampled = wav_pcm::linear_resample(raw_audio, wav_sr, sr_model);
    std::fprintf(stderr, "  batch resampled: %zu samples at %d Hz\n", batch_resampled.size(), sr_model);

    // Simulate streaming resampling: chunk-by-chunk like add_audio_chunk does.
    constexpr double kSimChunkSec = 1.0;
    std::vector<float> streaming_resampled;
    {
      const size_t chunk_samples = static_cast<size_t>(
          std::max(1., kSimChunkSec * static_cast<double>(wav_sr)));
      size_t offset = 0;
      while (offset < raw_audio.size()) {
        const size_t n = std::min(chunk_samples, raw_audio.size() - offset);
        std::vector<float> chunk(raw_audio.data() + offset, raw_audio.data() + offset + n);
        std::vector<float> res = wav_pcm::linear_resample(chunk, wav_sr, sr_model);
        streaming_resampled.insert(streaming_resampled.end(), res.begin(), res.end());
        offset += n;
      }
    }
    std::fprintf(stderr, "  streaming resampled: %zu samples at %d Hz\n", streaming_resampled.size(), sr_model);
    compare_vectors(batch_resampled, streaming_resampled, "resampled_audio");

    // ---- Stage 1: Per-chunk segmentation ORT comparison ----
    std::fprintf(stderr, "\n--- Stage 1: Per-chunk segmentation comparison ---\n");
    std::fprintf(stderr, "  (using batch-resampled audio for both to isolate seg from resampling)\n");

    const int num_channels = engine.segmentation_num_channels();
    const int chunk_num_samples = engine.segmentation_chunk_num_samples();
    const double chunk_step_sec = engine.segmentation_chunk_step_sec();
    const int step_samples = static_cast<int>(
        std::lrint(chunk_step_sec * static_cast<double>(sr_model)));

    auto count_chunks = [&](int64_t num_samples) -> std::pair<int64_t, bool> {
      int64_t complete = 0;
      if (num_samples >= chunk_num_samples) {
        complete = (num_samples - chunk_num_samples) / step_samples + 1;
      }
      bool has_last = (num_samples < chunk_num_samples) ||
                      ((num_samples - chunk_num_samples) % step_samples > 0);
      return {complete + (has_last ? 1 : 0), has_last};
    };

    auto [batch_chunks, batch_has_last] = count_chunks(static_cast<int64_t>(batch_resampled.size()));
    auto [stream_chunks, stream_has_last] = count_chunks(static_cast<int64_t>(streaming_resampled.size()));
    std::fprintf(stderr, "  batch chunks: %lld (has_last=%d)  streaming chunks: %lld (has_last=%d)\n",
                 (long long)batch_chunks, batch_has_last, (long long)stream_chunks, stream_has_last);

    if (batch_chunks != stream_chunks) {
      std::fprintf(stderr, "  CHUNK COUNT MISMATCH — this alone causes accuracy differences\n");
    }

    // Run segmentation on both resampled buffers and compare.
    const int64_t C = batch_chunks;
    const int64_t batch_complete = batch_resampled.size() >= static_cast<size_t>(chunk_num_samples)
        ? (static_cast<int64_t>(batch_resampled.size()) - chunk_num_samples) / step_samples + 1 : 0;

    // First chunk to discover F/K.
    {
      auto buf0 = pyannote::CppAnnote::extract_chunk_audio(
          batch_resampled.data(), static_cast<int64_t>(batch_resampled.size()),
          0, chunk_num_samples, num_channels);
      engine.run_segmentation_ort_single(buf0.data());
    }
    const int F = engine.seg_frames_per_chunk();
    const int K = engine.seg_classes();
    const int dim = engine.embedding_dimension();
    const int FK = F * K;

    std::fprintf(stderr, "  F=%d  K=%d  dim=%d\n", F, K, dim);

    // Build full seg tensors for batch audio.
    std::vector<float> seg_batch(static_cast<size_t>(C) * static_cast<size_t>(FK));
    for (int64_t c = 0; c < C; ++c) {
      const int64_t off = (c < batch_complete) ? c * step_samples : batch_complete * step_samples;
      auto buf = pyannote::CppAnnote::extract_chunk_audio(
          batch_resampled.data(), static_cast<int64_t>(batch_resampled.size()),
          off, chunk_num_samples, num_channels);
      auto seg = engine.run_segmentation_ort_single(buf.data());
      std::memcpy(&seg_batch[static_cast<size_t>(c) * FK], seg.data(), FK * sizeof(float));
    }

    // Build full seg tensors for streaming-resampled audio (same chunk logic).
    const int64_t C_s = stream_chunks;
    const int64_t stream_complete = streaming_resampled.size() >= static_cast<size_t>(chunk_num_samples)
        ? (static_cast<int64_t>(streaming_resampled.size()) - chunk_num_samples) / step_samples + 1 : 0;

    std::vector<float> seg_stream(static_cast<size_t>(C_s) * static_cast<size_t>(FK));
    for (int64_t c = 0; c < C_s; ++c) {
      const int64_t off = (c < stream_complete) ? c * step_samples : stream_complete * step_samples;
      auto buf = pyannote::CppAnnote::extract_chunk_audio(
          streaming_resampled.data(), static_cast<int64_t>(streaming_resampled.size()),
          off, chunk_num_samples, num_channels);
      auto seg = engine.run_segmentation_ort_single(buf.data());
      std::memcpy(&seg_stream[static_cast<size_t>(c) * FK], seg.data(), FK * sizeof(float));
    }

    // Compare only the overlapping chunks.
    const int64_t C_cmp = std::min(C, C_s);
    std::vector<float> seg_b_sub(seg_batch.begin(), seg_batch.begin() + C_cmp * FK);
    std::vector<float> seg_s_sub(seg_stream.begin(), seg_stream.begin() + C_cmp * FK);
    compare_vectors(seg_b_sub, seg_s_sub, "segmentation");

    // Per-chunk segmentation diff breakdown.
    int chunks_with_seg_diff = 0;
    for (int64_t c = 0; c < C_cmp; ++c) {
      bool any_diff = false;
      for (int i = 0; i < FK; ++i) {
        if (seg_b_sub[c * FK + i] != seg_s_sub[c * FK + i]) {
          any_diff = true;
          break;
        }
      }
      if (any_diff) ++chunks_with_seg_diff;
    }
    std::fprintf(stderr, "  chunks with any seg diff: %d / %lld\n", chunks_with_seg_diff, (long long)C_cmp);

    // ---- Stage 2: Per-chunk embedding comparison ----
    std::fprintf(stderr, "\n--- Stage 2: Per-chunk embedding comparison ---\n");
    std::fprintf(stderr, "  (using respective resampled audio + segmentation for each path)\n");

    std::vector<float> emb_batch(
        static_cast<size_t>(C) * K * dim, std::numeric_limits<float>::quiet_NaN());
    for (int64_t c = 0; c < C; ++c) {
      const int64_t off = (c < batch_complete) ? c * step_samples : batch_complete * step_samples;
      auto mono = pyannote::CppAnnote::extract_chunk_audio(
          batch_resampled.data(), static_cast<int64_t>(batch_resampled.size()),
          off, chunk_num_samples, 1);
      auto emb_c = engine.run_embedding_ort_single(
          mono.data(), &seg_batch[static_cast<size_t>(c) * FK]);
      std::memcpy(&emb_batch[static_cast<size_t>(c) * K * dim], emb_c.data(), K * dim * sizeof(float));
    }

    std::vector<float> emb_stream(
        static_cast<size_t>(C_s) * K * dim, std::numeric_limits<float>::quiet_NaN());
    for (int64_t c = 0; c < C_s; ++c) {
      const int64_t off = (c < stream_complete) ? c * step_samples : stream_complete * step_samples;
      auto mono = pyannote::CppAnnote::extract_chunk_audio(
          streaming_resampled.data(), static_cast<int64_t>(streaming_resampled.size()),
          off, chunk_num_samples, 1);
      auto emb_c = engine.run_embedding_ort_single(
          mono.data(), &seg_stream[static_cast<size_t>(c) * FK]);
      std::memcpy(&emb_stream[static_cast<size_t>(c) * K * dim], emb_c.data(), K * dim * sizeof(float));
    }

    std::vector<float> emb_b_sub(emb_batch.begin(), emb_batch.begin() + C_cmp * K * dim);
    std::vector<float> emb_s_sub(emb_stream.begin(), emb_stream.begin() + C_cmp * K * dim);
    compare_vectors(emb_b_sub, emb_s_sub, "embeddings");

    // ---- Stage 3: Cluster and decode comparison ----
    std::fprintf(stderr, "\n--- Stage 3: cluster_and_decode comparison ---\n");

    pyannote::DiarizationProfile prof_b, prof_s;
    auto turns_batch = engine.cluster_and_decode(seg_batch, emb_batch, static_cast<int>(C), prof_b);
    auto turns_stream = engine.cluster_and_decode(seg_stream, emb_stream, static_cast<int>(C_s), prof_s);

    compare_turns(turns_batch, turns_stream);

    // ---- Stage 4: Full end-to-end comparison via actual streaming session ----
    std::fprintf(stderr, "\n--- Stage 4: Full end-to-end (actual batch vs actual streaming) ---\n");

    pyannote::DiarizationProfile batch_prof;
    auto batch_turns = engine.diarize_mono_model_sr(
        std::vector<float>(batch_resampled), batch_prof);

    pyannote::StreamingDiarizationConfig cfg;
    cfg.refresh_every_sec = refresh_every_sec;
    pyannote::StreamingDiarizationSession sess(engine, cfg);
    sess.start_session();
    const size_t feed_samples = static_cast<size_t>(
        std::max(1., kSimChunkSec * static_cast<double>(wav_sr)));
    size_t off = 0;
    while (off < raw_audio.size()) {
      const size_t n = std::min(feed_samples, raw_audio.size() - off);
      sess.add_audio_chunk(raw_audio.data() + off, n, wav_sr);
      off += n;
    }
    auto snap = sess.end_session();
    std::vector<pyannote::DiarizationTurn> stream_turns;
    stream_turns.reserve(snap.turns.size());
    for (const auto& t : snap.turns) {
      stream_turns.push_back(pyannote::DiarizationTurn{t.start, t.end, t.speaker});
    }

    compare_turns(batch_turns, stream_turns);

    std::fprintf(stderr, "\n=== DONE ===\n");

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
