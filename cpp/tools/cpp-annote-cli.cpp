// SPDX-License-Identifier: MIT
// CLI for short-path diarization; core logic lives in ``src/cpp-annote.h`` (class ``pyannote::CppAnnote``).

#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpp-annote.h"
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

struct DiarJob {
  std::string wav;
  /// Optional per-utterance ``golden_speaker_bounds.json``; empty uses constructor default.
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
    if (cols.size() == 1) {
      if (out_dir.empty()) {
        throw std::runtime_error("manifest " + manifest_path + " line " + std::to_string(lineno) +
                                 ": single-column lines require --out-dir (wav path only)");
      }
      const fs::path wv(cols[0]);
      jobs.push_back({cols[0], "", (fs::path(out_dir) / (wv.stem().string() + ".json")).string()});
    } else if (cols.size() == 2) {
      jobs.push_back({cols[0], "", cols[1]});
    } else if (cols.size() == 3) {
      jobs.push_back({cols[0], cols[1], cols[2]});
    } else {
      throw std::runtime_error("manifest " + manifest_path + " line " + std::to_string(lineno) +
                               ": expected 1, 2, or 3 tab-separated fields (wav | wav,out | wav,bounds,out)");
    }
  }
  if (jobs.empty()) {
    throw std::runtime_error("manifest has no data rows: " + manifest_path);
  }
  return jobs;
}

static std::vector<DiarJob> load_wav_list_jobs(const std::string& list_path, const fs::path& out_dir) {
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
    jobs.push_back({line, "", (out_dir / (stem + ".json")).string()});
  }
  if (jobs.empty()) {
    throw std::runtime_error("wav-list has no paths: " + list_path);
  }
  return jobs;
}

int main(int argc, char** argv) {
  if (argc < 2 || has_flag(argc, argv, "--help")) {
    std::cerr
        << "cpp-annote-cli — WAV + segmentation ORT + ORT embedding + VBx → diarization JSON.\n\n"
        << "Required for every run:\n"
        << "  --segmentation-onnx PATH   (metadata: same stem .json)\n"
        << "  --embedding-onnx PATH      community1-embedding.onnx (+ same-stem .json)\n\n"
        << "Single file:\n"
        << "  --wav PATH\n"
        << "  --out PATH                 output diarization.json\n\n"
        << "Batch — tab-separated manifest (one job per line, # comments OK):\n"
        << "  --manifest PATH\n"
        << "    1 field:   wav   (requires --out-dir → OUT/<wav_stem>.json)\n"
        << "    2 fields:  wav<TAB>out.json\n"
        << "    3 fields:  wav<TAB>golden_speaker_bounds.json<TAB>out.json\n"
        << "  --out-dir PATH             required for 1-column manifest lines; also for --wav-list\n\n"
        << "Batch — one WAV path per line:\n"
        << "  --wav-list PATH            requires --out-dir; writes OUT/<stem>.json per line\n\n"
        << "Optional overrides (defaults compiled into the binary from export_cpp_annote_embedded.py):\n"
        << "  --receptive-field PATH         receptive_field.json\n"
        << "  --pipeline-snapshot PATH       pipeline_snapshot.json\n"
        << "  --golden-speaker-bounds PATH   default max_speakers cap when a job omits per-utterance bounds\n"
        << "  --xvec-transform PATH          xvec_transform.npz (must pair with --plda)\n"
        << "  --plda PATH                    plda.npz\n"
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
    const std::string global_bounds = get_arg(argc, argv, "--golden-speaker-bounds");
    const std::string snap = get_arg(argc, argv, "--pipeline-snapshot");
    const std::string embed_onnx = get_arg(argc, argv, "--embedding-onnx");
    const std::string xvec_npz = get_arg(argc, argv, "--xvec-transform");
    const std::string plda_npz = get_arg(argc, argv, "--plda");
    const bool continue_on_error = has_flag(argc, argv, "--continue-on-error");

    if (embed_onnx.empty()) {
      throw std::runtime_error("missing --embedding-onnx (see --help)");
    }
    if ((!xvec_npz.empty() && plda_npz.empty()) || (xvec_npz.empty() && !plda_npz.empty())) {
      throw std::runtime_error("provide both --xvec-transform and --plda, or neither for embedded weights");
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
      if (out_dir.empty()) {
        throw std::runtime_error("--wav-list requires --out-dir");
      }
      jobs = load_wav_list_jobs(wav_list_path, fs::path(out_dir));
    } else {
      const std::string out_path = get_arg(argc, argv, "--out");
      if (wav_path.empty() || out_path.empty()) {
        throw std::runtime_error("missing --wav or --out (see --help)");
      }
      if (!out_dir.empty()) {
        throw std::runtime_error("--out-dir is only for --manifest or --wav-list batch mode");
      }
      jobs.push_back({wav_path, "", out_path});
    }

    if (onnx_path.empty()) {
      throw std::runtime_error("missing --segmentation-onnx");
    }

    pyannote::CppAnnote engine(
        onnx_path,
        rf_path,
        global_bounds,
        snap,
        embed_onnx,
        xvec_npz,
        plda_npz);

    int n_fail = 0;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      try {
        const DiarJob& job = jobs[i];
        if (!job.golden_bounds.empty()) {
          engine.set_golden_speaker_bounds(job.golden_bounds);
        } else {
          engine.set_golden_speaker_bounds("");
        }

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
