// SPDX-License-Identifier: MIT
// Short-path diarization: WAV → full-chunk segmentation ORT → speaker count →
// reconstruct + to_diarization + to_annotation, using oracle hard_clusters from NPZ
// (same shape as Python dump). No VBx / PLDA / embeddings.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <climits>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "annotation_support.hpp"
#include "cnpy.h"
#include "wav_pcm_float32.hpp"

namespace fs = std::filesystem;

static std::string read_text(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    throw std::runtime_error("open failed: " + p);
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static double json_double(const std::string& json, const char* key) {
  const std::string pat = std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("json missing \"") + key + "\"");
  }
  return std::stod(m[1].str());
}

static bool json_bool(const std::string& json, const char* key) {
  const std::string pat = std::string("\"") + key + "\"\\s*:\\s*(true|false)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    throw std::runtime_error(std::string("json missing bool \"") + key + "\"");
  }
  return m[1].str() == "true";
}

static int closest_frame(double t, double sw_start, double sw_duration, double sw_step) {
  const double x = (t - sw_start - 0.5 * sw_duration) / sw_step;
  return static_cast<int>(std::lrint(x));
}

static void trim_warmup_inplace(
    std::vector<float>& data,
    size_t num_chunks,
    size_t& num_frames,
    size_t num_classes,
    double warm0,
    double warm1,
    double& chunk_start,
    double& chunk_duration) {
  const size_t n_left = static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm0));
  const size_t n_right = static_cast<size_t>(std::lrint(static_cast<double>(num_frames) * warm1));
  if (n_left + n_right >= num_frames) {
    throw std::runtime_error("trim: warm_up removes all frames");
  }
  const size_t new_frames = num_frames - n_left - n_right;
  std::vector<float> out(num_chunks * new_frames * num_classes);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < new_frames; ++f) {
      for (size_t k = 0; k < num_classes; ++k) {
        out[(c * new_frames + f) * num_classes + k] =
            data[(c * num_frames + (f + n_left)) * num_classes + k];
      }
    }
  }
  data.swap(out);
  num_frames = new_frames;
  chunk_start += warm0 * chunk_duration;
  chunk_duration *= (1.0 - warm0 - warm1);
}

static void inference_aggregate(
    const std::vector<float>& scores,
    size_t num_chunks,
    size_t num_frames_per_chunk,
    size_t num_classes,
    double chunks_start,
    double chunks_duration,
    double chunks_step,
    double out_duration,
    double out_step,
    bool skip_average,
    float epsilon,
    float missing,
    std::vector<float>& out_avg,
    int& num_out_frames) {
  const double out_sw_start = chunks_start;
  const double out_sw_duration = out_duration;
  const double out_sw_step = out_step;
  const double end_t = chunks_start + chunks_duration +
                       static_cast<double>(num_chunks - 1) * chunks_step + 0.5 * out_sw_duration;
  num_out_frames = closest_frame(end_t, out_sw_start, out_sw_duration, out_sw_step) + 1;
  if (num_out_frames <= 0) {
    throw std::runtime_error("aggregate: non-positive num_out_frames");
  }
  const int nf = static_cast<int>(num_frames_per_chunk);
  const int nc = static_cast<int>(num_classes);
  std::vector<float> agg(static_cast<size_t>(num_out_frames) * num_classes, 0.f);
  std::vector<float> occ(static_cast<size_t>(num_out_frames) * num_classes, 0.f);
  std::vector<float> mask_max(static_cast<size_t>(num_out_frames) * num_classes, 0.f);
  for (size_t ci = 0; ci < num_chunks; ++ci) {
    const double chunk_start = chunks_start + static_cast<double>(ci) * chunks_step;
    const int start_frame =
        closest_frame(chunk_start + 0.5 * out_sw_duration, out_sw_start, out_sw_duration, out_sw_step);
    for (int j = 0; j < nf; ++j) {
      for (int k = 0; k < nc; ++k) {
        const size_t idx =
            (ci * static_cast<size_t>(nf) + static_cast<size_t>(j)) * num_classes + static_cast<size_t>(k);
        const float raw = scores[idx];
        const float mask = std::isnan(raw) ? 0.f : 1.f;
        float score = raw;
        if (std::isnan(score)) {
          score = 0.f;
        }
        const int fi = start_frame + j;
        if (fi < 0 || fi >= num_out_frames) {
          continue;
        }
        const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) + static_cast<size_t>(k);
        agg[o] += score * mask;
        occ[o] += mask;
        mask_max[o] = std::max(mask_max[o], mask);
      }
    }
  }
  out_avg.resize(static_cast<size_t>(num_out_frames) * num_classes);
  for (int fi = 0; fi < num_out_frames; ++fi) {
    for (int k = 0; k < nc; ++k) {
      const size_t o = static_cast<size_t>(fi) * static_cast<size_t>(nc) + static_cast<size_t>(k);
      if (skip_average) {
        out_avg[o] = agg[o];
      } else {
        out_avg[o] = agg[o] / std::max(occ[o], epsilon);
      }
      if (mask_max[o] == 0.f) {
        out_avg[o] = missing;
      }
    }
  }
}

static std::vector<std::uint8_t> speaker_count_initial_uint8(
    std::vector<float> binarized,  // (C,F,L) copy
    size_t num_chunks,
    size_t num_frames,
    size_t num_classes,
    double& chunk_start,
    double chunk_step,
    double& chunk_duration,
    double rf_dur,
    double rf_step,
    int& num_out_frames) {
  size_t nf = num_frames;
  trim_warmup_inplace(binarized, num_chunks, nf, num_classes, 0.0, 0.0, chunk_start, chunk_duration);
  std::vector<float> summed(num_chunks * nf * 1);
  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t f = 0; f < nf; ++f) {
      float s = 0.f;
      for (size_t k = 0; k < num_classes; ++k) {
        s += binarized[(c * nf + f) * num_classes + k];
      }
      summed[c * nf + f] = s;
    }
  }
  std::vector<float> avg;
  inference_aggregate(
      summed,
      num_chunks,
      nf,
      1,
      chunk_start,
      chunk_duration,
      chunk_step,
      rf_dur,
      rf_step,
      false,
      1e-12f,
      0.f,
      avg,
      num_out_frames);
  std::vector<std::uint8_t> out(static_cast<size_t>(num_out_frames));
  for (int i = 0; i < num_out_frames; ++i) {
    const double r = std::rint(static_cast<double>(avg[static_cast<size_t>(i)]));
    out[static_cast<size_t>(i)] =
        static_cast<std::uint8_t>(std::max(0.0, std::min(255.0, r)));
  }
  return out;
}

static std::vector<std::int8_t> cap_count(
    const std::vector<std::uint8_t>& u8, int max_cap) {
  std::vector<std::int8_t> out(u8.size());
  for (size_t i = 0; i < u8.size(); ++i) {
    const int m = std::min(static_cast<int>(u8[i]), max_cap);
    out[i] = static_cast<std::int8_t>(static_cast<std::uint8_t>(m));
  }
  return out;
}

static void crop_loose_frame_range(
    double focus_start,
    double focus_end,
    double sw_start,
    double sw_duration,
    double sw_step,
    int& out_i0,
    int& out_i1_exclusive) {
  const double i_ = (focus_start - sw_duration - sw_start) / sw_step;
  int i = static_cast<int>(std::ceil(i_));
  const double j_ = (focus_end - sw_start) / sw_step;
  int j = static_cast<int>(std::floor(j_));
  out_i0 = i;
  out_i1_exclusive = j + 1;
}

static void extent_of_frames(
    double sw_start, double sw_step, size_t n_rows, double& seg_start, double& seg_end) {
  seg_start = sw_start;
  seg_end = sw_start + static_cast<double>(n_rows) * sw_step;
}

static void crop_feature_loose(
    const std::vector<float>& data,
    int n_samples,
    int n_cols,
    double sw_start,
    double sw_duration,
    double sw_step,
    double focus_start,
    double focus_end,
    std::vector<float>& out_data,
    int& out_rows,
    double& new_sw_start) {
  int i0 = 0;
  int i1 = 0;
  crop_loose_frame_range(focus_start, focus_end, sw_start, sw_duration, sw_step, i0, i1);
  const int clipped0 = std::max(0, i0);
  const int clipped1 = std::min(n_samples, i1);
  if (clipped0 >= clipped1) {
    out_rows = 0;
    out_data.clear();
    new_sw_start = sw_start;
    return;
  }
  out_rows = clipped1 - clipped0;
  out_data.resize(static_cast<size_t>(out_rows) * static_cast<size_t>(n_cols));
  for (int r = 0; r < out_rows; ++r) {
    const int src_row = clipped0 + r;
    for (int c = 0; c < n_cols; ++c) {
      out_data[static_cast<size_t>(r) * static_cast<size_t>(n_cols) + static_cast<size_t>(c)] =
          data[static_cast<size_t>(src_row) * static_cast<size_t>(n_cols) + static_cast<size_t>(c)];
    }
  }
  new_sw_start = sw_start + static_cast<double>(clipped0) * sw_step;
}

static std::vector<int> argsort_desc_stable(const float* row, int k) {
  std::vector<int> idx(static_cast<size_t>(k));
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [row](int a, int b) {
    if (row[a] != row[b]) {
      return row[a] > row[b];
    }
    return a < b;
  });
  return idx;
}

static std::vector<float> reconstruct_to_diarization(
    const std::vector<float>& segmentations,
    int C,
    int F,
    int L,
    double seg_ss,
    double seg_sd,
    double seg_st,
    const std::int8_t* hard_clusters,
    const std::vector<std::int8_t>& count_flat,
    double cnt_ss,
    double cnt_sd,
    double cnt_st,
    int& out_num_speaker_cols) {
  int max_clu = -3;
  for (int c = 0; c < C; ++c) {
    for (int j = 0; j < L; ++j) {
      max_clu = std::max(max_clu, static_cast<int>(hard_clusters[c * L + j]));
    }
  }
  const int num_clusters = max_clu + 1;
  if (num_clusters <= 0) {
    throw std::runtime_error("reconstruct: no positive cluster ids");
  }
  std::vector<float> clustered(static_cast<size_t>(C) * static_cast<size_t>(F) * static_cast<size_t>(num_clusters),
                             std::numeric_limits<float>::quiet_NaN());
  for (int c = 0; c < C; ++c) {
    const float* segm = &segmentations[static_cast<size_t>(c) * static_cast<size_t>(F) * static_cast<size_t>(L)];
    const std::int8_t* cluster = &hard_clusters[c * L];
    std::vector<char> seen_k(static_cast<size_t>(num_clusters), 0);
    for (int j = 0; j < L; ++j) {
      const int k = static_cast<int>(cluster[j]);
      if (k == -2) {
        continue;
      }
      if (k >= 0 && k < num_clusters) {
        seen_k[static_cast<size_t>(k)] = 1;
      }
    }
    for (int k = 0; k < num_clusters; ++k) {
      if (!seen_k[static_cast<size_t>(k)]) {
        continue;
      }
      for (int f = 0; f < F; ++f) {
        bool any_finite = false;
        float m = 0.f;
        for (int j = 0; j < L; ++j) {
          if (static_cast<int>(cluster[j]) != k) {
            continue;
          }
          const float v = segm[static_cast<size_t>(f) * static_cast<size_t>(L) + static_cast<size_t>(j)];
          if (std::isnan(v)) {
            continue;
          }
          if (!any_finite) {
            m = v;
            any_finite = true;
          } else {
            m = std::max(m, v);
          }
        }
        const size_t dst = (static_cast<size_t>(c) * static_cast<size_t>(F) + static_cast<size_t>(f)) *
                               static_cast<size_t>(num_clusters) +
                           static_cast<size_t>(k);
        clustered[dst] = any_finite ? m : std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  std::vector<float> activations;
  int T = 0;
  inference_aggregate(
      clustered,
      static_cast<size_t>(C),
      static_cast<size_t>(F),
      static_cast<size_t>(num_clusters),
      seg_ss,
      seg_sd,
      seg_st,
      cnt_sd,
      cnt_st,
      true,
      1e-12f,
      0.f,
      activations,
      T);
  int K = num_clusters;
  int max_spf = 0;
  for (size_t t = 0; t < count_flat.size(); ++t) {
    max_spf = std::max(max_spf, static_cast<int>(count_flat[t]));
  }
  max_spf = std::max(0, max_spf);
  if (K < max_spf) {
    std::vector<float> padded(static_cast<size_t>(T) * static_cast<size_t>(max_spf), 0.f);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        padded[static_cast<size_t>(t) * static_cast<size_t>(max_spf) + static_cast<size_t>(k)] =
            activations[static_cast<size_t>(t) * static_cast<size_t>(K) + static_cast<size_t>(k)];
      }
    }
    activations.swap(padded);
    K = max_spf;
  }
  double act_s = 0, act_e = 0;
  extent_of_frames(cnt_ss, cnt_st, static_cast<size_t>(T), act_s, act_e);
  const int Tcnt = static_cast<int>(count_flat.size());
  double cnt_s = 0, cnt_e = 0;
  extent_of_frames(cnt_ss, cnt_st, static_cast<size_t>(Tcnt), cnt_s, cnt_e);
  const double inter_s = std::max(act_s, cnt_s);
  const double inter_e = std::min(act_e, cnt_e);
  std::vector<float> act_cropped;
  int act_rows = 0;
  double tmp0 = 0.;
  crop_feature_loose(activations, T, K, cnt_ss, cnt_sd, cnt_st, inter_s, inter_e, act_cropped, act_rows, tmp0);
  std::vector<float> cnt_2d(static_cast<size_t>(Tcnt));
  for (int t = 0; t < Tcnt; ++t) {
    cnt_2d[static_cast<size_t>(t)] = static_cast<float>(count_flat[static_cast<size_t>(t)]);
  }
  std::vector<float> cnt_cropped;
  int cnt_rows = 0;
  double tmp1 = 0.;
  crop_feature_loose(cnt_2d, Tcnt, 1, cnt_ss, cnt_sd, cnt_st, inter_s, inter_e, cnt_cropped, cnt_rows, tmp1);
  if (act_rows != cnt_rows) {
    throw std::runtime_error("crop row mismatch");
  }
  std::vector<float> binary(static_cast<size_t>(act_rows) * static_cast<size_t>(K), 0.f);
  for (int t = 0; t < act_rows; ++t) {
    const float* arow = &act_cropped[static_cast<size_t>(t) * static_cast<size_t>(K)];
    std::vector<int> order = argsort_desc_stable(arow, K);
    const int c = static_cast<int>(std::lrint(static_cast<double>(cnt_cropped[static_cast<size_t>(t)])));
    const int c_use = std::max(0, std::min(c, K));
    for (int i = 0; i < c_use; ++i) {
      binary[static_cast<size_t>(t) * static_cast<size_t>(K) + static_cast<size_t>(order[static_cast<size_t>(i)])] =
          1.f;
    }
  }
  out_num_speaker_cols = K;
  return binary;
}

static void binarize_column(
    const float* k_scores,
    int num_frames,
    double sw_start,
    double sw_dur,
    double sw_step,
    double onset,
    double offset,
    double pad_onset,
    double pad_offset,
    std::vector<std::pair<double, double>>& regions_out) {
  if (num_frames <= 0) {
    return;
  }
  std::vector<double> ts(static_cast<size_t>(num_frames));
  for (int i = 0; i < num_frames; ++i) {
    ts[static_cast<size_t>(i)] = sw_start + static_cast<double>(i) * sw_step + 0.5 * sw_dur;
  }
  double start = ts[0];
  bool is_active = static_cast<double>(k_scores[0]) > onset;
  for (int i = 1; i < num_frames; ++i) {
    const double t = ts[static_cast<size_t>(i)];
    const double y = static_cast<double>(k_scores[i]);
    if (is_active) {
      if (y < offset) {
        regions_out.emplace_back(start - pad_onset, t + pad_offset);
        start = t;
        is_active = false;
      }
    } else {
      if (y > onset) {
        start = t;
        is_active = true;
      }
    }
  }
  if (is_active) {
    const double t_end = ts[static_cast<size_t>(num_frames - 1)];
    regions_out.emplace_back(start - pad_onset, t_end + pad_offset);
  }
}

static std::map<int, std::string> parse_label_mapping(const std::string& json) {
  std::map<int, std::string> m;
  std::regex re("\"([0-9]+)\"\\s*:\\s*\"([^\"]*)\"");
  for (std::sregex_iterator it(json.begin(), json.end(), re), end; it != end; ++it) {
    m[std::stoi((*it)[1].str())] = (*it)[2].str();
  }
  if (m.empty()) {
    throw std::runtime_error("label_mapping: empty");
  }
  return m;
}

static bool try_regex_double(const std::string& json, const std::string& key_esc, double& out) {
  const std::string pat = "\"" + key_esc + "\"\\s*:\\s*([-+0-9.eE]+)";
  std::regex re(pat);
  std::smatch m;
  if (!std::regex_search(json, m, re)) {
    return false;
  }
  out = std::stod(m[1].str());
  return true;
}

static void filter_min_duration_on(std::vector<std::pair<double, double>>& regs, double min_on) {
  if (min_on <= 0.0) {
    return;
  }
  std::vector<std::pair<double, double>> kept;
  for (const auto& pr : regs) {
    if (pr.second - pr.first >= min_on - 1e-12) {
      kept.push_back(pr);
    }
  }
  regs.swap(kept);
}

struct Turn {
  double start = 0.;
  double end = 0.;
  std::string speaker;
  bool operator<(const Turn& o) const {
    if (start != o.start) {
      return start < o.start;
    }
    if (end != o.end) {
      return end < o.end;
    }
    return speaker < o.speaker;
  }
};

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += c;
  }
  return o;
}

static void write_diarization_json(const std::string& path, const std::vector<Turn>& turns) {
  std::ofstream f(path);
  if (!f) {
    throw std::runtime_error("write failed: " + path);
  }
  f << std::setprecision(17);
  f << "[\n";
  for (size_t i = 0; i < turns.size(); ++i) {
    const Turn& t = turns[i];
    f << "  {\"start\": " << t.start << ", \"end\": " << t.end << ", \"speaker\": \"" << json_escape(t.speaker)
      << "\"}";
    if (i + 1 < turns.size()) {
      f << ",";
    }
    f << "\n";
  }
  f << "]\n";
}

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

struct DiarJob {
  std::string wav;
  std::string clusters;
  std::string label_mapping;
  std::string golden_bounds;  // empty → use global_bounds in run_diar_job
  std::string out;
};

struct SegConfig {
  int sr_model = 0;
  int num_channels = 0;
  int chunk_num_samples = 0;
  bool multilabel_export = false;
  double chunk_step_sec = 0.;
  double chunk_dur_sec = 0.;
};

static void trim_inplace(std::string& s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
    s.erase(s.begin());
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
    s.pop_back();
  }
}

static std::vector<std::string> split_tab_row(const std::string& line) {
  std::vector<std::string> cols;
  size_t i = 0;
  while (i < line.size()) {
    const size_t j = line.find('\t', i);
    if (j == std::string::npos) {
      cols.push_back(line.substr(i));
      break;
    }
    cols.push_back(line.substr(i, j - i));
    i = j + 1;
  }
  for (auto& c : cols) {
    trim_inplace(c);
  }
  return cols;
}

static std::vector<DiarJob> load_manifest_jobs(const std::string& manifest_path, const std::string& out_dir) {
  std::vector<DiarJob> jobs;
  std::ifstream f(manifest_path);
  if (!f) {
    throw std::runtime_error("open manifest failed: " + manifest_path);
  }
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    trim_inplace(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    auto cols = split_tab_row(line);
    while (!cols.empty() && cols.back().empty()) {
      cols.pop_back();
    }
    if (cols.size() == 3) {
      if (out_dir.empty()) {
        throw std::runtime_error("manifest line " + std::to_string(lineno) +
                                 ": three columns require --out-dir (wav, clusters, label_mapping)");
      }
      const fs::path wv(cols[0]);
      jobs.push_back(
          {cols[0], cols[1], cols[2], "", (fs::path(out_dir) / (wv.stem().string() + ".json")).string()});
    } else if (cols.size() == 4) {
      jobs.push_back({cols[0], cols[1], cols[2], "", cols[3]});
    } else if (cols.size() == 5) {
      jobs.push_back({cols[0], cols[1], cols[2], cols[3], cols[4]});
    } else {
      throw std::runtime_error("manifest " + manifest_path + " line " + std::to_string(lineno) +
                               ": expected 3, 4, or 5 tab-separated fields");
    }
  }
  if (jobs.empty()) {
    throw std::runtime_error("manifest has no data rows: " + manifest_path);
  }
  return jobs;
}

static std::vector<DiarJob> load_wav_list_jobs(
    const std::string& list_path, const fs::path& artifact_base, const fs::path& out_dir) {
  std::vector<DiarJob> jobs;
  std::ifstream f(list_path);
  if (!f) {
    throw std::runtime_error("open wav-list failed: " + list_path);
  }
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    trim_inplace(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const fs::path wv(line);
    const std::string stem = wv.stem().string();
    const fs::path sub = artifact_base / stem;
    DiarJob j;
    j.wav = line;
    j.clusters = (sub / "hard_clusters_final.npz").string();
    j.label_mapping = (sub / "label_mapping.json").string();
    const fs::path gb = sub / "golden_speaker_bounds.json";
    if (fs::exists(gb)) {
      j.golden_bounds = gb.string();
    }
    j.out = (out_dir / (stem + ".json")).string();
    jobs.push_back(std::move(j));
  }
  if (jobs.empty()) {
    throw std::runtime_error("wav-list has no paths: " + list_path);
  }
  return jobs;
}

static void run_diar_job(
    Ort::Session& session,
    Ort::MemoryInfo& mem,
    const char* in_name_ptr,
    const char* out_name_ptr,
    const SegConfig& cfg,
    const DiarJob& job,
    const std::string& rf_path,
    const std::string& global_bounds,
    double min_off,
    double min_on) {
  const fs::path outp(job.out);
  const fs::path parent = outp.parent_path();
  if (!parent.empty()) {
    fs::create_directories(parent);
  }

  const int sr_model = cfg.sr_model;
  const int num_channels = cfg.num_channels;
  const int chunk_num_samples = cfg.chunk_num_samples;
  const bool multilabel_export = cfg.multilabel_export;
  const double chunk_step_sec = cfg.chunk_step_sec;
  const double chunk_dur_sec = cfg.chunk_dur_sec;

  int wav_sr = 0;
  std::vector<float> mono = wav_pcm::load_wav_pcm16_mono_float32(job.wav, wav_sr);
  std::vector<float> audio = wav_pcm::linear_resample(mono, wav_sr, sr_model);

  const int step_samples = static_cast<int>(std::lrint(chunk_step_sec * static_cast<double>(sr_model)));
  if (step_samples <= 0 || chunk_num_samples <= 0) {
    throw std::runtime_error("bad chunk/step samples");
  }

  const int64_t num_samples = static_cast<int64_t>(audio.size());
  int64_t num_chunks = 0;
  if (num_samples >= chunk_num_samples) {
    num_chunks = (num_samples - chunk_num_samples) / step_samples + 1;
  }
  const bool has_last =
      (num_samples < chunk_num_samples) || ((num_samples - chunk_num_samples) % step_samples > 0);
  const int64_t total_chunks = num_chunks + (has_last ? 1 : 0);
  if (total_chunks <= 0) {
    write_diarization_json(job.out, {});
    std::cout << "Wrote empty diarization (audio shorter than one chunk) → " << job.out << "\n";
    return;
  }

  cnpy::npz_t clz = cnpy::npz_load(job.clusters);
  if (!clz.count("hard_clusters")) {
    throw std::runtime_error("clusters npz missing hard_clusters: " + job.clusters);
  }
  const cnpy::NpyArray& hc = clz["hard_clusters"];
  if (hc.shape.size() != 2 || static_cast<int64_t>(hc.shape[0]) != total_chunks) {
    std::ostringstream oss;
    oss << "hard_clusters first dim " << hc.shape[0] << " != num_chunks " << total_chunks << " (" << job.wav
        << ")";
    throw std::runtime_error(oss.str());
  }
  const std::int8_t* hptr = hc.data<std::int8_t>();

  std::vector<float> seg_out;
  int n_frames = 0;
  int n_classes = 0;

  auto run_chunk = [&](const float* wav_data, int64_t chunk_idx) {
    const std::array<int64_t, 3> in_shape{1, num_channels, chunk_num_samples};
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(wav_data),
        static_cast<size_t>(1 * num_channels * chunk_num_samples),
        in_shape.data(),
        in_shape.size());
    const char* in_names[] = {in_name_ptr};
    const char* out_names[] = {out_name_ptr};
    auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
    float* op = outs[0].GetTensorMutableData<float>();
    auto sh = outs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (sh.size() != 3 || sh[0] != 1) {
      throw std::runtime_error("unexpected ORT output rank");
    }
    const int F = static_cast<int>(sh[1]);
    const int K = static_cast<int>(sh[2]);
    if (chunk_idx == 0) {
      n_frames = F;
      n_classes = K;
      seg_out.resize(static_cast<size_t>(total_chunks) * static_cast<size_t>(F) * static_cast<size_t>(K));
    } else if (F != n_frames || K != n_classes) {
      throw std::runtime_error("ORT output shape changed across chunks");
    }
    std::memcpy(
        &seg_out[static_cast<size_t>(chunk_idx) * static_cast<size_t>(F) * static_cast<size_t>(K)],
        op,
        static_cast<size_t>(F * K) * sizeof(float));
  };

  for (int64_t c = 0; c < num_chunks; ++c) {
    const int64_t off = c * step_samples;
    std::vector<float> buf(static_cast<size_t>(num_channels) * static_cast<size_t>(chunk_num_samples), 0.f);
    for (int ch = 0; ch < num_channels; ++ch) {
      for (int i = 0; i < chunk_num_samples; ++i) {
        const int64_t si = off + i;
        float v = 0.f;
        if (si >= 0 && si < num_samples) {
          v = audio[static_cast<size_t>(si)];
        }
        buf[static_cast<size_t>(ch) * static_cast<size_t>(chunk_num_samples) + static_cast<size_t>(i)] = v;
      }
    }
    run_chunk(buf.data(), c);
  }
  if (has_last) {
    const int64_t off = num_chunks * step_samples;
    std::vector<float> buf(static_cast<size_t>(num_channels) * static_cast<size_t>(chunk_num_samples), 0.f);
    for (int ch = 0; ch < num_channels; ++ch) {
      for (int i = 0; i < chunk_num_samples; ++i) {
        const int64_t si = off + i;
        float v = 0.f;
        if (si >= 0 && si < num_samples) {
          v = audio[static_cast<size_t>(si)];
        }
        buf[static_cast<size_t>(ch) * static_cast<size_t>(chunk_num_samples) + static_cast<size_t>(i)] = v;
      }
    }
    run_chunk(buf.data(), num_chunks);
  }

  const int C = static_cast<int>(total_chunks);
  const int F = n_frames;
  const int Kcls = n_classes;
  std::vector<float> binarized = seg_out;
  if (!multilabel_export) {
    throw std::runtime_error("shortpath currently expects export_includes_powerset_to_multilabel=true");
  }

  double seg_ss = 0.;
  double seg_sd = chunk_dur_sec;
  double seg_st = chunk_step_sec;
  const std::string rf_txt = read_text(rf_path);
  const double rf_dur = json_double(rf_txt, "duration");
  const double rf_step = json_double(rf_txt, "step");

  int Tcnt = 0;
  std::vector<std::uint8_t> cnt_u8 = speaker_count_initial_uint8(
      binarized,
      static_cast<size_t>(C),
      static_cast<size_t>(F),
      static_cast<size_t>(Kcls),
      seg_ss,
      seg_st,
      seg_sd,
      rf_dur,
      rf_step,
      Tcnt);
  std::uint8_t max_cnt = 0;
  for (std::uint8_t v : cnt_u8) {
    max_cnt = std::max(max_cnt, v);
  }
  if (max_cnt == 0) {
    write_diarization_json(job.out, {});
    std::cout << "Wrote empty diarization (no speech / zero speaker count), chunks=" << C << " → " << job.out
              << "\n";
    return;
  }
  int max_cap = INT_MAX;
  const std::string bounds_path = !job.golden_bounds.empty() ? job.golden_bounds : global_bounds;
  if (!bounds_path.empty()) {
    const std::string bj = read_text(bounds_path);
    std::regex re("\"max_speakers\"\\s*:\\s*(null|[-+0-9]+)");
    std::smatch m;
    if (std::regex_search(bj, m, re) && m[1].str() != "null") {
      max_cap = std::stoi(m[1].str());
    }
  }
  std::vector<std::int8_t> count_i8 = cap_count(cnt_u8, max_cap);

  const double onset = 0.5;
  const double offset = 0.5;
  const double pad_onset = 0.0;
  const double pad_offset = 0.0;
  const bool apply_annotation_support = (min_off > 0.0 || pad_onset > 0.0 || pad_offset > 0.0);

  int K_di = 0;
  const std::vector<float> discrete = reconstruct_to_diarization(
      seg_out, C, F, Kcls, seg_ss, seg_sd, seg_st, hptr, count_i8, 0.0, rf_dur, rf_step, K_di);
  if (K_di <= 0) {
    throw std::runtime_error("reconstruct returned non-positive speaker column count");
  }
  const int rows = static_cast<int>(discrete.size() / static_cast<size_t>(K_di));

  const std::map<int, std::string> label_map = parse_label_mapping(read_text(job.label_mapping));
  for (int k = 0; k < K_di; ++k) {
    if (!label_map.count(k)) {
      throw std::runtime_error("label_mapping missing key " + std::to_string(k) + " (have 0.." +
                               std::to_string(K_di - 1) + ")");
    }
  }
  std::map<int, std::vector<pyannote_port::Segment>> by_label;
  for (int k = 0; k < K_di; ++k) {
    std::vector<float> col(static_cast<size_t>(rows));
    for (int t = 0; t < rows; ++t) {
      col[static_cast<size_t>(t)] =
          discrete[static_cast<size_t>(t) * static_cast<size_t>(K_di) + static_cast<size_t>(k)];
    }
    std::vector<std::pair<double, double>> regs;
    binarize_column(
        col.data(),
        rows,
        0.0,
        rf_dur,
        rf_step,
        onset,
        offset,
        pad_onset,
        pad_offset,
        regs);
    if (!apply_annotation_support) {
      filter_min_duration_on(regs, min_on);
    }
    for (const auto& pr : regs) {
      by_label[k].push_back(pyannote_port::Segment{pr.first, pr.second});
    }
  }

  std::vector<Turn> turns;
  if (apply_annotation_support) {
    const auto merged = pyannote_port::annotation_support(by_label, min_off);
    for (const auto& pr : merged) {
      const pyannote_port::Segment& seg = pr.second;
      if (seg.duration() < min_on - 1e-12) {
        continue;
      }
      turns.push_back({seg.start, seg.end, label_map.at(pr.first)});
    }
  } else {
    for (int k = 0; k < K_di; ++k) {
      for (const pyannote_port::Segment& seg : by_label[k]) {
        turns.push_back({seg.start, seg.end, label_map.at(k)});
      }
    }
  }
  std::sort(turns.begin(), turns.end());
  write_diarization_json(job.out, turns);
  std::cout << "Wrote " << job.out << " (" << turns.size() << " turns), chunks=" << C << "\n";
}

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help")) {
    std::cerr
        << "community1_shortpath — WAV + oracle clusters → diarization JSON (no VBx).\n\n"
        << "Single file (required):\n"
        << "  --wav PATH\n"
        << "  --segmentation-onnx PATH   (metadata: same stem .json)\n"
        << "  --receptive-field PATH     receptive_field.json\n"
        << "  --clusters PATH            hard_clusters_final.npz\n"
        << "  --label-mapping PATH       label_mapping.json\n"
        << "  --out PATH                 output diarization.json\n\n"
        << "Batch — tab-separated manifest (one job per line, # comments OK):\n"
        << "  --manifest PATH\n"
        << "    Line with 3 fields:  wav<TAB>clusters.npz<TAB>label_mapping.json  (needs --out-dir)\n"
        << "    Line with 4 fields:  wav<TAB>clusters<TAB>label_mapping<TAB>out.json\n"
        << "    Line with 5 fields:  wav<TAB>clusters<TAB>label_mapping<TAB>golden_speaker_bounds<TAB>out.json\n"
        << "  --out-dir PATH             required for 3-column manifest lines\n\n"
        << "Batch — one WAV path per line (same layout as Python golden dirs):\n"
        << "  --wav-list PATH            list file; each line is a .wav path\n"
        << "  --artifact-base DIR        use DIR/<wav_stem>/hard_clusters_final.npz, label_mapping.json,\n"
        << "                             and golden_speaker_bounds.json if that file exists\n"
        << "  --out-dir PATH             write DIR/<wav_stem>.json for each line\n\n"
        << "Optional (all modes):\n"
        << "  --golden-speaker-bounds PATH   used when a job has no per-utterance bounds (manifest/wav-list)\n"
        << "  --pipeline-snapshot PATH       pipeline_snapshot.json (segmentation.min_duration_*)\n"
        << "  --continue-on-error            batch only: print error and continue; exit 1 if any failed\n";
    return 2;
  }
  try {
    const std::string onnx_path = get_arg(argc, argv, "--segmentation-onnx");
    const std::string rf_path = get_arg(argc, argv, "--receptive-field");
    const std::string manifest_path = get_arg(argc, argv, "--manifest");
    const std::string wav_list_path = get_arg(argc, argv, "--wav-list");
    const std::string wav_path = get_arg(argc, argv, "--wav");
    const std::string out_dir = get_arg(argc, argv, "--out-dir");
    const std::string artifact_base_str = get_arg(argc, argv, "--artifact-base");
    const std::string global_bounds = get_arg(argc, argv, "--golden-speaker-bounds");
    const std::string snap = get_arg(argc, argv, "--pipeline-snapshot");
    const bool continue_on_error = has_flag(argc, argv, "--continue-on-error");

    double min_off = 0.;
    double min_on = 0.;
    if (!snap.empty()) {
      const std::string sj = read_text(snap);
      try_regex_double(sj, "segmentation\\.min_duration_off", min_off);
      try_regex_double(sj, "segmentation\\.min_duration_on", min_on);
    }

    std::vector<DiarJob> jobs;
    if (!manifest_path.empty()) {
      if (!wav_list_path.empty() || !wav_path.empty()) {
        throw std::runtime_error("use only one of --manifest, --wav-list, or --wav");
      }
      jobs = load_manifest_jobs(manifest_path, out_dir);
    } else if (!wav_list_path.empty()) {
      if (!wav_path.empty()) {
        throw std::runtime_error("use only one of --manifest, --wav-list, or --wav");
      }
      if (artifact_base_str.empty() || out_dir.empty()) {
        throw std::runtime_error("--wav-list requires --artifact-base and --out-dir");
      }
      jobs = load_wav_list_jobs(wav_list_path, fs::path(artifact_base_str), fs::path(out_dir));
    } else {
      const std::string clu_path = get_arg(argc, argv, "--clusters");
      const std::string map_path = get_arg(argc, argv, "--label-mapping");
      const std::string out_path = get_arg(argc, argv, "--out");
      if (wav_path.empty() || onnx_path.empty() || rf_path.empty() || clu_path.empty() || map_path.empty() ||
          out_path.empty()) {
        throw std::runtime_error("missing required argument (see --help)");
      }
      if (!out_dir.empty()) {
        throw std::runtime_error("--out-dir is only for --manifest or --wav-list batch mode");
      }
      jobs.push_back({wav_path, clu_path, map_path, "", out_path});
    }

    if (onnx_path.empty() || rf_path.empty()) {
      throw std::runtime_error("missing --segmentation-onnx or --receptive-field");
    }

    std::string json_path = onnx_path;
    if (json_path.size() > 5 && json_path.substr(json_path.size() - 5) == ".onnx") {
      json_path = json_path.substr(0, json_path.size() - 5) + ".json";
    } else {
      json_path += ".json";
    }
    const std::string seg_json = read_text(json_path);
    SegConfig cfg;
    cfg.sr_model = static_cast<int>(json_double(seg_json, "sample_rate"));
    cfg.num_channels = static_cast<int>(json_double(seg_json, "num_channels"));
    cfg.chunk_num_samples = static_cast<int>(json_double(seg_json, "chunk_num_samples"));
    cfg.multilabel_export = json_bool(seg_json, "export_includes_powerset_to_multilabel");
    cfg.chunk_step_sec = 0.1 * json_double(seg_json, "chunk_duration_sec");
    {
      std::regex re("\"chunk_step_sec\"\\s*:\\s*([-+0-9.eE]+)");
      std::smatch m;
      if (std::regex_search(seg_json, m, re)) {
        cfg.chunk_step_sec = std::stod(m[1].str());
      }
    }
    cfg.chunk_dur_sec = json_double(seg_json, "chunk_duration_sec");

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "community1_shortpath");
    Ort::SessionOptions opts;
    Ort::Session session(env, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
    Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, alloc);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int n_fail = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      try {
        run_diar_job(
            session,
            mem,
            in_name.get(),
            out_name.get(),
            cfg,
            jobs[i],
            rf_path,
            global_bounds,
            min_off,
            min_on);
      } catch (const std::exception& e) {
        std::cerr << "ERROR";
        if (jobs.size() > 1) {
          std::cerr << " [" << (i + 1) << "/" << jobs.size() << "] " << jobs[i].wav;
        }
        std::cerr << ": " << e.what() << "\n";
        ++n_fail;
        if (!continue_on_error) {
          return 1;
        }
      }
    }
    if (n_fail > 0) {
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
