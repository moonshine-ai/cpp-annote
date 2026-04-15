// SPDX-License-Identifier: MIT
// CLI for short-path diarization; core logic lives in ``port/pyannote.hpp`` (class ``pyannote::Pyannote``).

#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyannote.hpp"
#include "wav_pcm_float32.hpp"

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

struct DiarJob {
  std::string wav;
  std::string clusters;
  std::string label_mapping;
  std::string golden_bounds;
  std::string out;
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

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help")) {
    std::cerr
        << "community1_shortpath — WAV + segmentation ORT → diarization JSON "
           "(oracle clusters and/or full VBx; see flags).\n\n"
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
        << "  --artifact-base BASE        use BASE/<wav_stem>/hard_clusters_final.npz, label_mapping.json,\n"
        << "                             and golden_speaker_bounds.json if present\n"
        << "  --out-dir PATH             write OUT/<wav_stem>.json for each line\n\n"
        << "Full C++ pipeline (segmentation + ORT embedding + VBx; no oracle clusters NPZ):\n"
        << "  --embedding-onnx PATH      community1-embedding.onnx (+ same-stem .json)\n"
        << "  --xvec-transform PATH      xvec_transform.npz\n"
        << "  --plda PATH                plda.npz\n"
        << "  When all three are set, --clusters / manifest clusters column are ignored.\n\n"
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
    const std::string embed_onnx = get_arg(argc, argv, "--embedding-onnx");
    const std::string xvec_npz = get_arg(argc, argv, "--xvec-transform");
    const std::string plda_npz = get_arg(argc, argv, "--plda");
    const bool continue_on_error = has_flag(argc, argv, "--continue-on-error");
    const bool vbx_cli = !embed_onnx.empty() && !xvec_npz.empty() && !plda_npz.empty();

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
      if (wav_path.empty() || onnx_path.empty() || rf_path.empty() || map_path.empty() || out_path.empty()) {
        throw std::runtime_error("missing required argument (see --help)");
      }
      if (!vbx_cli && clu_path.empty()) {
        throw std::runtime_error("missing --clusters (or pass --embedding-onnx, --xvec-transform, --plda)");
      }
      if (!out_dir.empty()) {
        throw std::runtime_error("--out-dir is only for --manifest or --wav-list batch mode");
      }
      jobs.push_back({wav_path, clu_path, map_path, "", out_path});
    }

    if (onnx_path.empty() || rf_path.empty()) {
      throw std::runtime_error("missing --segmentation-onnx or --receptive-field");
    }

    pyannote::Pyannote engine(
        onnx_path,
        rf_path,
        vbx_cli ? std::string() : jobs[0].clusters,
        jobs[0].label_mapping,
        global_bounds,
        snap,
        embed_onnx,
        xvec_npz,
        plda_npz);

    int n_fail = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      try {
        const DiarJob& job = jobs[i];
        engine.set_utterance_paths(job.clusters, job.label_mapping, job.golden_bounds);

        int wav_sr = 0;
        std::vector<float> mono = wav_pcm::load_wav_pcm16_mono_float32(job.wav, wav_sr);
        const fs::path outp(job.out);
        const fs::path parent = outp.parent_path();
        if (!parent.empty()) {
          fs::create_directories(parent);
        }
        std::vector<pyannote::DiarizationTurn> turns =
            engine.diarize(std::move(mono), static_cast<std::int32_t>(wav_sr));
        pyannote::write_diarization_json(job.out, turns);
        if (turns.empty()) {
          std::cout << "Wrote empty diarization → " << job.out << "\n";
        } else {
          std::cout << "Wrote " << job.out << " (" << turns.size() << " turns)\n";
        }
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
