# Embedding and VBx parity vs Python

This document captures how the C++ diarization path in **`cpp-annote-cli`** (ORT embeddings + ported VBx) can diverge from Python (Torch embeddings + `VBxClustering`), and a **staged plan** to close the gap using **logging and tests at every boundary**.

Related: `cpp/porting-plan.md` (Section 11, golden artifact layout), `cpp/scripts/dump_diarization_golden.py`, `cpp/scripts/write_vbx_golden_reference.py`, `cpp/scripts/trace_diarization_parity.py`.

---

## Two independent gaps

1. **Embeddings (Torch vs ORT)**  
   Golden dumps store `embeddings.npz` from the **PyTorch** `get_embeddings` path. C++ full mode uses **fbank + ONNX**. Differences in fbank, framing, weights, or ORT numerics appear here first.

2. **VBx stack (even on identical embeddings)**  
   `clustering_golden_test` loads Python `embeddings.npz` and compares C++ `hard_clusters` to golden `clustering.npz` with a **relaxed** threshold (`mismatch_frac > 0.35` fails). That reflects known non–bit-identical behavior (SciPy/sklearn vs Eigen + Hungarian + `scipy_linkage` port, ELBO early stop, ties).

**Implication:** large DER differences in C++ **cpp-annote-cli** vs Python are expected until both gaps are addressed. Rebuilding alone does not make the C++ path match Python.

---

## Stage map (Python → C++)

| Stage | Python | C++ |
|--------|--------|-----|
| Binarized segmentations | `binarized_segmentations.npz` from dump | Same NPZ as input to clustering |
| Filtered training rows | `BaseClustering.filter_embeddings` in `src/pyannote/audio/pipelines/clustering.py` | `filter_embeddings` in `cpp/port/clustering_vbx.cpp` |
| AHC | `scipy.linkage` + `scipy.cluster.hierarchy.fcluster` | `scipy_linkage::*` in `cpp/port/scipy_linkage.hpp` |
| PLDA features | `self.plda(train_embeddings)` | `PldaModel::operator()` in `cpp/port/plda_vbx.cpp` (aligned with `vbx_setup` in `src/pyannote/audio/utils/vbx.py`) |
| VBx | `cluster_vbx` → `VBx` in `vbx.py` (`maxIters=20`, `epsilon=1e-4`, `init_smoothing=7.0`) | `plda_vbx::cluster_vbx` (default `elbo_epsilon=1e-4`) |
| Centroids / soft | `cdist` cosine, `soft = 2 - distance` | `cdist_cosine`, same formula |
| Assignment | `scipy.optimize.linear_sum_assignment(..., maximize=True)` | `hungarian::min_cost_assignment` |
| Inactive speakers | `hard_clusters[...] = -2` | Same in `clustering_vbx.cpp` |

### Known divergence risks

- **Single-speaker mask:** Python uses `np.sum(segmentations.data, axis=2, keepdims=True) == 1`. C++ uses `|row_sum - 1| < 1e-3`. Equivalent for strict binary masks; log and compare `clean` counts if segmentations are ever soft.
- **VBx iteration count:** ELBO early stop can differ → different `gamma` → different centroids → different hard labels.
- **Hungarian:** multiple optimal assignments on ties → different `hard_clusters` with identical soft scores.
- **Forced cluster count:** Python `KMeans(..., random_state=42)` vs C++ `kmeans_fit_predict(..., seed=42)` when `num_clusters` / bounds force relabeling; ensure `VbxClusteringParams` in `cpp-annote.cpp` match the resolved pipeline config from the golden `pipeline_snapshot.json` / HF `config.yaml`.

---

## Logging strategy

### Single control knob (implemented)

| Variable | Meaning |
|----------|---------|
| `PYANNOTE_CPP_PARITY` unset / `0` | Off (default). |
| `PYANNOTE_CPP_PARITY=1` | Light: one line after ORT embeddings, one after VBx (or skip line if `T < 2`). |
| `PYANNOTE_CPP_PARITY=2` | Heavy: writes `vbx_parity_dump.npz` under `PYANNOTE_CPP_PARITY_OUT` (requires that directory). |

Implementation: `cpp/port/parity_log.hpp`, `parity_log.cpp`; hooks in `cpp-annote.cpp` (embeddings) and `clustering_vbx.cpp` (VBx + dump).

`PYANNOTE_CPP_DIAG=1` remains separate (chunking / ORT shape only).

### Level 1 (light)

Per utterance, **no large tensors**:

- **Embeddings:** `C`, `Kcls`, `dim`, NaN count, `fp=` FNV fingerprint over strided float32 samples (`parity::fingerprint_float32`, stride 409).  
- **VBx:** `T_train`, `Kvb`, `max_hard_cluster`, `vbx_iters`, `vbx_last_elbo_delta`, `fea_fp=` same fingerprint on PLDA `fea` (row-major float32 cast).

Prefix: `[PYANNOTE_CPP_PARITY]`.

Python helper for the same fingerprint on a golden `.npz` array: `cpp/scripts/parity_fingerprint.py <file.npz> <key>`.

### Level 2 (heavy)

When `PYANNOTE_CPP_PARITY=2` and `PYANNOTE_CPP_PARITY_OUT` is set, creates the directory if needed and writes **`$PYANNOTE_CPP_PARITY_OUT/vbx_parity_dump.npz`** with:

| Key | Description |
|-----|-------------|
| `chunk_idx`, `spk_idx` | int32, filter indices |
| `train`, `train_n` | float64, shapes `(T, dim)` row-major |
| `pdist_condensed`, `linkage_Z` | float64, SciPy-style linkage flat `Z` |
| `ahc` | int32 |
| `fea`, `Phi` | float64 |
| `gamma`, `pi` | float64 (VBx outputs) |
| `centroids`, `soft_clusters` | float64 |
| `hard_clusters` | int32 (C·S vector) |

Use the same utterance + golden inputs to compare against Python dumps (`write_vbx_golden_reference.py` etc.).

---

## Testing milestones (order matters)

Each milestone should have an explicit **pass criterion** before tightening the next.

### Milestone 0 — Frozen inputs

One fixed golden utterance directory (e.g. from `dump_diarization_golden.py` or `cpp/golden/...`). Assert NPZ keys, shapes, dtypes.

**Implemented:** `cpp/tests/frozen_golden_inputs_test.cpp` → **`cpp/build/frozen_golden_inputs_test`**. CMake sets default paths to `golden/callhome_eng_idx0/callhome_eng_data_idx0_head120s` and parent `golden/callhome_eng_idx0`. Pass an utterance directory as `argv[1]` to override. If you regenerate the bundle, update the `constexpr` expectations in that source file and re-run.

### Milestone 1 — Embedding parity (Torch vs ORT)

- **Chunk 0 / spk 0:** `cpp/tests/embedding_golden_test.cpp` (ORT vs `embedding_chunk0_spk0_ort.npz` from the dump).  
- **Full tensor:** same binary, mode `--all`: recomputes ORT embeddings for every chunk and local speaker using the same fbank + **segmentation-length** `weights` tensor as `CppAnnote::diarize` (`embedding_ort_infer.cpp` + `compute_fbank.cpp`), WAV + golden `binarized_segmentations.npz`, parent `pipeline_snapshot.json` for `embedding_exclude_overlap`, and compares to `embeddings.npz`.

Printed stats: slot counts (`both_finite`, `gold_finite_ort_nan`, …), mean of per-slot max-abs-diff over **mutually finite** rows, worst `(c,s)`, global max abs diff, `allclose` fail fraction (same `rtol`/`atol` as chunk-0 mode).

**Pass gates (defaults are pragmatic for the checked-in CallHome slice; tighten with env):**

| Env | Default | Role |
|-----|---------|------|
| `EMBEDDING_FULL_MAX_OR_NAN_SLOTS` | `200` | Max slots where golden row is finite but ORT is all-NaN (should stay `0` once weights-length parity holds). Use `0` to fail on any such slot. |
| `EMBEDDING_FULL_MAX_FAIL_FRAC` | `0.10` | Max fraction of mutually-finite slots that violate `allclose(rtol=1e-2, atol=1e-3)`. |

Shared ORT helpers live in **`cpp/port/embedding_ort_infer.hpp`** / **`embedding_ort_infer.cpp`** (also used by `cpp-annote-cli` / `cpp-annote.cpp`).

#### Investigating Torch-finite / ORT-NaN slots

**Resolved root cause:** Golden **`embeddings.npz`** comes from **`get_embeddings` → `pipeline._embedding(...)`** with **Torch WeSpeaker** (`model_(waveforms, weights=masks)`), i.e. **`resnet(fbank, weights=…)`** where **`weights`** has **segmentation length `F`** (same as `embedding_chunk0_spk0_ort.npz`). The exported ONNX graph accepts **`weights` with time axis `F`** and resizes internally to fbank length.

C++ had been **upsampling** the mask to **`Tf`** with a non-PyTorch grid, then **gating** on **`n_keep < min_num_frames`** and skipping ORT like **`ONNXWeSpeakerPretrainedSpeakerEmbedding.__call__`** (strip short masked fbank, leave NaN). That **does not** match the golden tensor when a speaker has **no active segmentation frames** (`n_active_seg = 0`): Torch still returns a **finite** pooled vector; ORT with **all-zero segmentation `weights`** also returns **finite** (verified).

**Fix (in `CppAnnote::diarize` and `embedding_golden_test --all`):** pass **`src`** (length **`F`**, clean vs full per `min_nf_seg`) directly to **`run_embedding_ort`**; remove the **`min_num_frames`** skip. **`seg_to_fbank_nearest_index`** is now only a helper (PyTorch/OpenCV-BC `nearest_neighbor_compute_source_index`); it is **not** used on the ORT path anymore.

**Tools:**

| Step | What to run |
|------|-------------|
| Global thresholds | `embedding_golden_test --all` prints `min_num_samples`, `min_num_frames`, `F`, `chunk_num_samples`, `embedding_exclude_overlap` (stderr). |
| Per-slot CSV | Set **`EMBEDDING_FULL_DUMP_CSV=/path/rows.csv`**. Writes one row per **gold-finite & ORT-NaN** slot: `c,s,n_active_seg,Tf,F,min_num_frames,min_nf_seg,sum_clean,prefer_clean,deficit` (`deficit = min_num_frames - n_active_seg`; diagnostic only). |
| Histogram | Same run prints **`[EMBEDDING_NAN_MISMATCH]`** only when such slots exist. |
| Python cross-check | **`cpp/scripts/embedding_slot_metrics.py`** recomputes `sum_clean`, `prefer_clean`, `n_active_seg` from `binarized_segmentations.npz` + CSV columns. |

**Earlier code fix (still applies):** `min_nf_seg` uses the **embedding-domain** chunk length in samples (`chunk_for_fbank.size()` after resample), matching Python’s `duration * self._embedding.sample_rate` denominator when rates differ.

**Execution notes (CallHome `idx0_head120s`, community-1 ONNX, after weights fix):**

- **`embedding_golden_test --all`:** `gold_finite_ort_nan=0`, `allclose_fail_slots=0`, mean max-abs-diff on mutually finite slots **~3e-6** (ORT vs golden).
- The previous **125** `gold_finite_ort_nan` rows were **inactive speakers** (`n_active_seg = 0` on the used mask) while golden stayed finite; ORT-on-full-chunk with **zero segmentation weights** matches golden once **`weights`** length **`F`** is wired correctly.

#### CallHome `idx3_head120s` — DER 29.64% (Python golden) vs 48.08% (C++ full)

Golden dir: ``.callhome_eval_work/golden/callhome_eng_data_idx3_head120s/`` (same layout as idx0). WAV: ``.callhome_eval_work/wavs/callhome_eng_data_idx3_head120s.wav``. Reference labels: HF ``diarizers-community/callhome`` ``eng``/``data`` row **3**, cropped to WAV length (120 s).

| Diagnostic | Command / artifact | Result |
|------------|--------------------|--------|
| DER ref vs ``diarization.json`` (Python dump) | ``pyannote.metrics.DiarizationErrorRate`` on HF ref vs golden JSON | **29.64%** |
| DER ref vs existing C++ full JSON | same vs ``.callhome_eval_work/cpp_out/callhome_eng_data_idx3_head120s.json`` | **48.08%** |
| DER ref vs reconstruct using **golden** ``hard_clusters_final.npz`` | ``reconstruct_golden_test`` / same reconstruct as Python dump | **29.64%** (matches Python golden to reported precision; not ``cpp-annote-cli``, which always runs VBx) |
| Full-window segmentation ORT vs golden | ``full_segmentation_window_parity_test`` … onnx … golden … wav | **PASS** — ``max_abs_diff = 0``, **0** binarized bit mismatches |
| First-chunk segmentation | ``segmentation_golden_test`` | **PASS** — ``max_abs_diff(first_chunk) = 0`` |
| Embedding ORT vs ``embeddings.npz`` | ``embedding_golden_test`` … ``--all`` … | **PASS** — ``gold_finite_ort_nan=0``, ``allclose_fail_slots=0``, mean per-slot max-abs **~2.6e-6** |
| VBx vs golden ``clustering.npz`` | ``clustering_golden_test`` on golden dir + HF ``xvec_transform.npz`` / ``plda.npz`` | **FAIL** — ``mismatch_frac ≈ 0.56`` (uses **golden** embeddings + binarized; compares C++ ``hard_clusters`` to Python ``clustering.npz``) |

**Conclusion:** Segmentation and ORT embeddings **match** the dumped Python tensors for this utterance. Reconstruction with Python’s golden ``hard_clusters`` reproduces the **same DER** as the golden diarization. The **~18.4 pt DER lift** on **cpp-annote-cli** output is therefore from **C++ VBx / AHC vs Python’s VBxClustering** (and downstream relabeling), not from waveform, segmentation ONNX, or embedding ONNX vs Torch golden.

**Next steps (Milestones 3–5):** align ``clustering_vbx`` with SciPy linkage / ``fcluster``, PLDA application, and VBx iterations (see traces in ``write_vbx_golden_reference.py``); ``clustering_golden_test`` already encodes the current acceptance gate (``mismatch_frac <= 0.35``) and fails on idx3 for the same reason as idx0 in a strict run.

### Milestone 2 — Filter parity

Python: run only `filter_embeddings` on golden `binarized_segmentations.npz` + `embeddings.npz`; save indices and `train_embeddings`.  
C++: CLI flag or test binary dumping the same from `filter_embeddings`.

**Pass:** identical `(chunk_idx, spk_idx)` and `max_abs_diff(train)` below a small tolerance (e.g. 1e-5).

### Milestone 3 — AHC parity

**Implemented**

- **`filter_train.hpp` / `filter_train.cpp`:** ``filter_embeddings_train`` matches Python ``VBxClustering.filter_embeddings`` (``segmentations.sum(axis=2)==1`` clean frames, ``num_clean_frames >= min_active_ratio * num_frames``, NaN filter). Used by ``clustering_vbx.cpp`` and tests.
- **`write_vbx_golden_reference.py`:** Uses the same NumPy filter as ``clustering.py``; writes ``train_chunk_idx``, ``train_speaker_idx``, ``fcluster_threshold`` into ``vbx_reference.npz`` alongside ``train_n``, ``pdist_condensed``, ``linkage_Z``, ``ahc``.
- **`filter_ahc_golden_test`:** ``filter_ahc_golden_test <vbx_reference.npz>`` checks ``pdist`` / centroid ``linkage_Z`` / ``fcluster`` + contiguous remap vs the NPZ (threshold from ``fcluster_threshold`` when present, else 0.6). With a second argument ``<utterance_dir>``, also runs the C++ filter on ``embeddings.npz`` + ``binarized_segmentations.npz`` and checks indices vs NPZ and L2-normalized ``train`` vs ``train_n``.
- **`linkage_fcluster_golden_test`:** Reads ``fcluster_threshold`` from the NPZ when present.

**Pass:** ``filter_ahc_golden_test`` exit **0** on the checked-in CallHome ``vbx_reference.npz`` (and with utterance dir for end-to-end filter + AHC). SciPy centroid linkage is already matched bit-for-bit on that reference (``pdist`` / ``Z`` tol ``1e-9``).

**Regenerate reference:** ``python cpp/scripts/write_vbx_golden_reference.py --golden-utterance … --xvec … --plda … --out …/vbx_reference.npz``

### Milestone 4 — PLDA `x0` / `fea` / `Phi`

Compare C++ `PldaModel` to Python `vbx_setup` on the same filtered `train` rows:

- **`x0`:** `xvec_tf(train)` (golden NPZ stores it as **C-contiguous** `float64` `(T, lda_dim)`).
- **`fea`:** `plda_tf(x0, lda_dim)` (same as `plda_tf(xvec_tf(train), …)`).
- **`Phi`:** first `lda_dim` entries of `plda_psi` after the generalized eigen reorder in `vbx_setup`.

**Implemented:** `cpp/tests/plda_fea_golden_test.cpp` → `cpp/build/plda_fea_golden_test`. Regenerate `vbx_reference.npz` with `write_vbx_golden_reference.py` (writes `train`, `x0`, `fea`, `Phi`, …).

**Pass:** max abs diff on `x0`, `fea`, and `Phi` each ≤ **1e-4** (current runs are ~**1e-12** on CallHome head slice + HF community-1 weights).

**Fixes that landed in `plda_vbx.cpp`:** NumPy-style **`(plda_tr.T / psi)`** divides **columns** by `psi[j]` (not rows). Load **`mean2` / `lda` as float32** from `xvec_transform.npz` when stored that way. For the HF community-1 `plda.npz`, align generalized eigenvector **columns** to SciPy’s sign using a fingerprinted `wccn[0, :]` row (Eigen vs LAPACK phase); other checkpoints skip alignment.

### Milestone 5 — VBx parity

- **Debug mode:** `elbo_epsilon < 0` (fixed `max_vbx_iters`, no early stop) on both sides; compare `gamma`, `pi` each iteration to traces from `write_vbx_golden_reference.py` (`gamma_trace`, `pi_trace`).  
- **Production mode:** keep `epsilon=1e-4`; assert same iteration count or bounded `gamma` drift.

**Pass:** with fixed iterations, `gamma` within tolerance; then validate early-stop alignment.

### Milestone 6 — Centroids and soft clusters

Compare Python and C++ `centroids` and per-chunk soft scores (or cosine distance before `2 - d`).

**Pass:** within tolerance before Hungarian.

### Milestone 7 — Hard clusters (Hungarian)

If soft matrices match but `hard_clusters` differ, investigate ties; consider **deterministic tie-breaking** in Hungarian or compare labels up to **global permutation** per chunk.

### Milestone 8 — Golden clustering test

Tighten `clustering_golden_test` from the current ~35% mismatch cap toward **exact** or **permutation-aware** equality once Milestones 2–7 pass.

### Milestone 9 — Full pipeline

`eval_callhome_cpp_vs_python.py` (C++ column uses **cpp-annote-cli**): DER should track Python only after Milestones **1** and **8** are acceptable for your product bar.

---

## CI recommendations

- **Fast:** embedding spot check + filter + PLDA on a tiny checked-in or generated fixture.  
- **Heavy (optional / nightly):** full utterance + CallHome head slice.  
- Use `SKIP_CLUSTERING_GOLDEN=1` only while Milestone 8 is red; remove or tighten once VBx aligns.

---

## Suggested fix order

1. **Embeddings** (Milestone 1) — largest lever for full-mode DER.  
2. **Filter + AHC** (Milestones 2–3) if embeddings match but VBx drifts.  
3. **VBx numerics** (Milestone 5) — fixed iterations + traces first.  
4. **Hungarian determinism** (Milestone 7) if soft match but hard labels differ.

---

## Quick reference: useful commands and artifacts

| Purpose | Location |
|---------|----------|
| Golden per-utterance bundle | `dump_diarization_golden.py` output layout in `porting-plan.md` §11 |
| VBx intermediate reference | `cpp/scripts/write_vbx_golden_reference.py` |
| ORT vs golden embedding | `cpp/build/embedding_golden_test` |
| VBx vs golden `hard_clusters` | `cpp/build/clustering_golden_test` |
| Trace / eval | `cpp/scripts/trace_diarization_parity.py`, `cpp/scripts/eval_callhome_cpp_vs_python.py` |
