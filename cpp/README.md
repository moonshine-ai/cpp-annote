# C++ bootstrap (ONNX Runtime)

Minimal **CMake** project that links **ONNX Runtime** + **cnpy** and runs:

- **`segmentation_golden_test`** — first chunk `waveforms` vs `segmentations.npz`
- **`embedding_golden_test`** — chunk0/spk0: `fbank` + `weights` vs `embedding_chunk0_spk0_ort.npz`; **`--all`**: ORT vs full `embeddings.npz` (Milestone 1)
- **`filter_ahc_golden_test`** — Milestone 3: C++ `filter_embeddings_train` + `pdist` / centroid `linkage` / `fcluster` vs `vbx_reference.npz` (optional second arg: utterance dir to cross-check filter indices and `train_n`)
- **`linkage_fcluster_golden_test`** — same AHC chain starting from stored `train_n` only (lighter)
- **`speaker_count_golden_test`** — `binarized_segmentations.npz` + `receptive_field.json` vs `speaker_count_initial.npz` (no ONNX; links **cnpy** only)
- **`frozen_golden_inputs_test`** — Milestone 0: asserts required keys, shapes, and dtypes in the frozen CallHome golden utterance NPZs (and sibling `receptive_field.json`, etc.); **cnpy** only
- **`reconstruct_golden_test`** — `reconstruct` + `to_diarization` vs `discrete_diarization_overlap.npz` and `discrete_diarization_exclusive.npz` (**cnpy** only)
- **`annotation_golden_test`** — `Binarize` / `to_annotation` vs `diarization.json` and `exclusive_diarization.json` (uses **`label_mapping.json`**; **cnpy** only)
- **`community1_shortpath`** — **WAV → full-chunk segmentation ORT → speaker count → reconstruct → JSON** using **oracle** `hard_clusters_final.npz` from a Python golden dump (same audio length / chunk count as that dump). No VBx, PLDA, or embeddings.
- **`pyannote::Pyannote`** (`port/pyannote.hpp`, `port/pyannote.cpp`) — same pipeline as **`community1_shortpath`**, exposed as a class: construct with paths to ONNX, receptive field, oracle clusters, label mapping, optional speaker bounds, optional pipeline snapshot; call **`diarize(std::vector<float> mono, std::int32_t sample_rate)`** to obtain **`std::vector<pyannote::DiarizationTurn>`** (use **`pyannote::write_diarization_json`** to serialize like the CLI).

## 1. Install ONNX Runtime (C/C++ prebuilt)

Download a release matching your OS/arch from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases) (e.g. `onnxruntime-osx-arm64-*.tgz` or `onnxruntime-linux-x64-*.tgz`), unpack, and set **`ONNXRUNTIME_ROOT`** to the unpacked root (headers under `include/onnxruntime/`, library under `lib/`).

**Homebrew (macOS):** often `export ONNXRUNTIME_ROOT=/opt/homebrew/opt/onnxruntime`.

```bash
export ONNXRUNTIME_ROOT="$HOME/local/onnxruntime-osx-arm64-1.18.1"
```

## 2. Golden bundle

Run (or **re-run**) the Python dumper so each utterance folder includes **`first_chunk_waveform.npz`**, **`embedding_chunk0_spk0_ort.npz`**, etc.:

```bash
./scripts/dump_golden.sh cpp/golden/my_run path/to/audio.wav
```

## 3. Configure and build

From the **repository root**:

```bash
./scripts/build_cpp.sh
```

Or manually:

```bash
cmake -S cpp -B cpp/build -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
cmake --build cpp/build --parallel
```

## 4. Run segmentation parity

```bash
./cpp/build/segmentation_golden_test \
  cpp/artifacts/community1-segmentation.onnx \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s
```

Exit code **0** means the first-chunk ORT output matches golden `segmentations.npz` within tolerances (see `segmentation_golden_test.cpp`).

### 5. Run embedding parity

```bash
./cpp/build/embedding_golden_test \
  cpp/artifacts/community1-embedding.onnx \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s
```

Requires **`embedding_chunk0_spk0_ort.npz`** in the utterance directory (added by `dump_diarization_golden.py`; re-run dump if missing).

### 5b. Full-tensor embedding parity (Milestone 1, optional)

Same WAV length as the golden dump (so chunk count matches `embeddings.npz`). Uses golden **`binarized_segmentations.npz`** and parent **`pipeline_snapshot.json`** for `embedding_exclude_overlap`.

```bash
./cpp/build/embedding_golden_test \
  cpp/artifacts/community1-embedding.onnx \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s \
  --all \
  path/to/same_audio_as_golden.wav \
  cpp/artifacts/community1-segmentation.onnx
```

Tolerances: **`EMBEDDING_FULL_MAX_OR_NAN_SLOTS`** (default `200`) and **`EMBEDDING_FULL_MAX_FAIL_FRAC`** (default `0.10`); see `cpp/embedding-vbx-parity-plan.md`. Use `0` / `0.02` when tightening parity.

Diagnostics: set **`EMBEDDING_FULL_DUMP_CSV=/path/rows.csv`** to write gold-finite / ORT-NaN slots with `n_active_seg` (count of used-mask segmentation frames above 0.5), `Tf`, `min_nf_seg`, `deficit`, etc.; stderr prints **`[EMBEDDING_NAN_MISMATCH]`** histograms when any such rows exist. Cross-check with:

```bash
python3 cpp/scripts/embedding_slot_metrics.py \
  --golden-dir cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s \
  --from-cpp-csv /path/rows.csv
```

### 6. Run speaker-count parity (post-net)

```bash
./cpp/build/speaker_count_golden_test \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s \
  cpp/golden/callhome_eng_idx0/receptive_field.json
```

The second path is the **`receptive_field.json`** next to the utterance folder in golden bundles produced by `dump_diarization_golden.py`. Exit code **0** means the recomputed count matches **`speaker_count_initial.npz`** exactly (after `rint` → `uint8`, as in Python).

If **`speaker_count_capped.npz`** is present, the same binary also checks the cap step (`np.minimum` then `int8`, as in the pipeline). Prefer **`golden_speaker_bounds.json`** in the utterance folder (written by the dumper) so `max_speakers` matches the bundle; if that file is missing, the test assumes **no cap** (`max_speakers = inf`), which matches bundles produced without `--max-speakers` when the capped tensor equals the initial one.

### 6b. Frozen golden inputs (Milestone 0)

Locks keys, ranks, shapes, and dtypes for the checked-in CallHome utterance bundle (see `cpp/embedding-vbx-parity-plan.md`). Re-run after regenerating golden NPZs; update the `constexpr` block in `cpp/tests/frozen_golden_inputs_test.cpp` when dimensions change.

```bash
./cpp/build/frozen_golden_inputs_test \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s
```

With no arguments, the binary uses CMake-defined default paths (same utterance).

### 7. Run reconstruct + `to_diarization` parity

```bash
./cpp/build/reconstruct_golden_test \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s
```

Requires **`segmentations.npz`**, **`hard_clusters_final.npz`**, **`speaker_count_capped.npz`**, and both discrete NPZs from a full (non–early-exit) golden dump. Exit code **0** means the C++ raster matches Python for overlap and exclusive branches.

### 8. Run `to_annotation` / Binarize parity (JSON)

```bash
./cpp/build/annotation_golden_test \
  cpp/golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s
```

Compares hysteresis (**onset/offset 0.5**, frame middles) on **`discrete_diarization_*.npz`** to **`diarization.json`** / **`exclusive_diarization.json`**, after **`label_mapping.json`** (same as the Python `rename_labels` step). Non-zero **`segmentation.min_duration_off`** in **`../pipeline_snapshot.json`** triggers the same **`Annotation.support(collar=...)`** merge as Python (see **`cpp/port/annotation_support.hpp`**). Non-zero **`min_duration_on`** drops short segments after that step. **`min_duration_off`** / pad branches match `Binarize.__call__`: the support pass runs only when **`min_duration_off > 0`** or pad is non-zero (Callhome bundle uses **0.0** / no pad).

### 9. Short path: WAV + oracle clusters → `diarization.json`

Use the **same** WAV file (and thus the same number of segmentation chunks) as the Python golden dump that produced **`hard_clusters_final.npz`**. The tool reads **PCM 16-bit LE** WAV, resamples linearly to the segmentation model rate from the ONNX sidecar JSON (typically **16 kHz** mono downmix).

```bash
./cpp/build/community1_shortpath \
  --wav /path/to/same_audio_as_golden.wav \
  --segmentation-onnx cpp/artifacts/community1-segmentation.onnx \
  --receptive-field cpp/golden/my_run/receptive_field.json \
  --clusters cpp/golden/my_run/utterance_dir/hard_clusters_final.npz \
  --label-mapping cpp/golden/my_run/utterance_dir/label_mapping.json \
  --golden-speaker-bounds cpp/golden/my_run/utterance_dir/golden_speaker_bounds.json \
  --pipeline-snapshot cpp/golden/my_run/pipeline_snapshot.json \
  --out /tmp/diarization_cpp.json
```

**`--golden-speaker-bounds`** and **`--pipeline-snapshot`** are optional; omit bounds to assume no `max_speakers` cap (`inf`). This build expects **`export_includes_powerset_to_multilabel: true`** (community-1 default export). There is **no** embedding / VBx / PLDA: clustering must be supplied out-of-band (oracle). If the segmentation model assigns **no** active speech (speaker count all zero), the tool writes an **empty** JSON array (same idea as Python early exit).

**Batch (same ONNX / receptive field / snapshot for every job):**

- **`--manifest FILE`** — tab-separated lines (lines starting with `#` are ignored). Use **paths relative to your cwd** (or absolute).
  - **3 columns:** `wav<TAB>clusters.npz<TAB>label_mapping.json` — also pass **`--out-dir DIR`**; each job writes `DIR/<wav_basename_without_suffix>.json`.
  - **4 columns:** `wav<TAB>clusters<TAB>label_mapping<TAB>out.json`
  - **5 columns:** `wav<TAB>clusters<TAB>label_mapping<TAB>golden_speaker_bounds.json<TAB>out.json`
- **`--wav-list FILE`** plus **`--artifact-base BASE`** and **`--out-dir OUT`** — one WAV path per line; artifacts are read from `BASE/<stem(wav)>/hard_clusters_final.npz`, `label_mapping.json`, and `golden_speaker_bounds.json` if present; outputs are `OUT/<stem(wav)>.json`.
- **`--continue-on-error`** — in batch mode, log each failure and keep going; exit status **1** if any job failed.

The binary loads the ONNX session once and reuses it across jobs.
