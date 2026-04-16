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
- **`cpp-annote-cli`** — **WAV → segmentation ORT → ORT embeddings → VBx (PLDA) → speaker count → reconstruct → JSON**. Cluster assignments are computed in C++ (no `hard_clusters_final.npz` from a golden dump).
- **`pyannote::CppAnnote`** (`src/cpp-annote.h`, `src/cpp-annote.cpp`, embedded defaults in **`src/community1_cpp_annote_embedded.*`**) — diarization engine: construct with segmentation ONNX and embedding ONNX; pass empty strings for receptive field, golden bounds, pipeline snapshot, and both PLDA NPZ paths to use compiled-in community-1 defaults, or pass file paths to override. Use via **`StreamingDiarizationSession`** (`src/cpp-annote-streaming.h`): call **`start_session()`**, feed audio with **`add_audio_chunk()`**, then **`end_session()`** to get **`StreamingDiarizationSnapshot`** with **`std::vector<StreamingDiarizationTurn>`**. Output speakers are **`SPEAKER_00`**, **`SPEAKER_01`**, … unless you extend the code.

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

Compares hysteresis (**onset/offset 0.5**, frame middles) on **`discrete_diarization_*.npz`** to **`diarization.json`** / **`exclusive_diarization.json`**, after **`label_mapping.json`** (same as the Python `rename_labels` step). Non-zero **`segmentation.min_duration_off`** in **`../pipeline_snapshot.json`** triggers the same **`Annotation.support(collar=...)`** merge as Python (see **`cpp/src/annotation_support.h`**). Non-zero **`min_duration_on`** drops short segments after that step. **`min_duration_off`** / pad branches match `Binarize.__call__`: the support pass runs only when **`min_duration_off > 0`** or pad is non-zero (Callhome bundle uses **0.0** / no pad).

### 9. Short path: WAV + VBx → `diarization.json`

The tool reads **PCM 16-bit LE** WAV, resamples to the segmentation model rate from the ONNX sidecar JSON (typically **16 kHz** mono downmix), runs **embedding ORT** on each chunk/local speaker, then **VBx** to produce `hard_clusters` in memory before reconstruct.

**Defaults compiled in:** `receptive_field.json`, `pipeline_snapshot.json`, `golden_speaker_bounds.json`, and the **`xvec_transform` / `plda`** tensors from the community-1 checkpoint are embedded as C++ arrays (`src/community1_cpp_annote_embedded.{h,cpp}`), regenerated by **`cpp/scripts/export_cpp_annote_embedded.py`** (also run automatically after **`export_segmentation_onnx.py`** / **`export_embedding_onnx.py`**).

```bash
./cpp/build/cpp-annote-cli \
  --wav /path/to/audio.wav \
  --segmentation-onnx cpp/artifacts/community1-segmentation.onnx \
  --embedding-onnx cpp/artifacts/community1-embedding.onnx \
  --out /tmp/diarization_cpp.json
```

Optional overrides: **`--receptive-field`**, **`--pipeline-snapshot`**, **`--golden-speaker-bounds`**, and **`--xvec-transform`** + **`--plda`** (both required together if you override NPZs). This build expects **`export_includes_powerset_to_multilabel: true`** (community-1 default export). If the segmentation model assigns **no** active speech (speaker count all zero), the tool writes an **empty** JSON array.

**Batch (same ONNX + embedding for every job; embedded JSON/PLDA unless overridden on the command line):**

- **`--manifest FILE`** — tab-separated lines (`#` comments OK). Paths relative to cwd or absolute.
  - **1 column:** `wav` — requires **`--out-dir DIR`**; each job writes `DIR/<wav_stem>.json`.
  - **2 columns:** `wav<TAB>out.json`
  - **3 columns:** `wav<TAB>golden_speaker_bounds.json<TAB>out.json`
- **`--wav-list FILE`** plus **`--out-dir OUT`** — one WAV path per line; outputs are `OUT/<stem>.json`.
- **`--continue-on-error`** — in batch mode, log each failure and keep going; exit **1** if any job failed.

The binary loads the segmentation and embedding ONNX sessions once and reuses them across jobs.
