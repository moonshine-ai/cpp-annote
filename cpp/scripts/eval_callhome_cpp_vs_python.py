#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Evaluate Diarization Error Rate (DER) on the first n clips from Hugging Face
# ``diarizers-community/callhome`` (English subset by default): reference labels
# from dataset timestamps vs (1) full ``SpeakerDiarization`` in Python and
# (2) C++ ``community1_shortpath`` either with oracle clusters from a golden dump
# (``--cpp-mode oracle``) or the full C++ stack including ORT embedding + VBx
# (``--cpp-mode full``, no ``hard_clusters_final.npz``).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

_SCRIPT_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for d in _SCRIPT_DIR.parents:
        if (d / "cpp" / "scripts" / "dump_diarization_golden.py").is_file():
            return d
    raise RuntimeError(
        "Could not locate pyannote-audio repository root "
        "(expected cpp/scripts/dump_diarization_golden.py in a parent of this script)."
    )


_REPO_ROOT = _repo_root()


def _pick_token(explicit: str | None) -> str | bool | None:
    if explicit:
        return explicit
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key)
        if v:
            return v
    return None


def _utterance_stem(subset: str, index: int, max_seconds: float) -> str:
    return f"callhome_{subset}_data_idx{index}_head{int(max_seconds)}s"


def _row_to_reference(row: dict[str, Any], crop_end: float, uri: str) -> Any:
    from pyannote.core import Annotation, Segment

    ann: Annotation = Annotation(uri=uri)
    for t0, t1, spk in zip(
        row["timestamps_start"],
        row["timestamps_end"],
        row["speakers"],
        strict=True,
    ):
        t0f = float(t0)
        t1f = float(t1)
        if t0f >= crop_end:
            continue
        t1f = min(t1f, crop_end)
        if t1f <= t0f:
            continue
        ann[Segment(t0f, t1f)] = str(spk)
    return ann


def _wav_duration_sec(path: Path) -> float:
    info = torchaudio.info(str(path))
    return float(info.num_frames) / float(info.sample_rate)


def _save_wav_row(row: dict[str, Any], path: Path, max_seconds: float) -> float:
    audio = row["audio"]
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    max_samples = int(max_seconds * sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
    wav = torch.from_numpy(arr).unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav, sr)
    return float(arr.shape[0]) / float(sr)


def _hf_embedding_plda_paths(
    checkpoint: str,
    revision: str | None,
    token: str | bool | None,
) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    kw: dict[str, Any] = {"repo_id": checkpoint, "token": token}
    if revision is not None:
        kw["revision"] = revision
    xvec = Path(hf_hub_download(filename="plda/xvec_transform.npz", **kw))
    plda = Path(hf_hub_download(filename="plda/plda.npz", **kw))
    return xvec, plda


def _json_diarization_to_annotation(path: Path, uri: str) -> Any:
    from pyannote.core import Annotation, Segment

    ann: Annotation = Annotation(uri=uri)
    with path.open(encoding="utf-8") as f:
        turns = json.load(f)
    for t in turns:
        ann[Segment(float(t["start"]), float(t["end"]))] = str(t["speaker"])
    return ann


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DER on first n CallHome (HF) clips: Python community-1 pipeline vs C++ shortpath "
            "(oracle clusters or full C++ VBx; see --cpp-mode)."
        )
    )
    ap.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=20,
        help="Number of consecutive dataset rows starting at --start-index (default: 20)",
    )
    ap.add_argument("--start-index", type=int, default=0, help="First row index in the split (default: 0)")
    ap.add_argument(
        "--subset",
        default="eng",
        choices=("eng", "zho", "deu", "jpn", "spa"),
        help="Callhome language subset (HF config name)",
    )
    ap.add_argument("--split", default="data", help="HF split name (default: data)")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=120.0,
        help="Truncate each call to this many seconds from the start (default: 120)",
    )
    ap.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / ".callhome_eval_work",
        help="Scratch directory for wavs, golden bundle, and C++ JSON output",
    )
    ap.add_argument(
        "--checkpoint",
        default="pyannote/speaker-diarization-community-1",
        help="Pipeline id for Python + golden dump (default: community-1)",
    )
    ap.add_argument("--revision", default=None)
    ap.add_argument("--token", default=None, help="HF token (else HF_TOKEN / HUGGING_FACE_HUB_TOKEN)")
    ap.add_argument(
        "--device",
        default="auto",
        help="Torch device for pipeline: auto | cpu | cuda | cuda:0 | mps | …",
    )
    ap.add_argument(
        "--cpp-binary",
        type=Path,
        default=_REPO_ROOT / "cpp" / "build" / "community1_shortpath",
        help="Path to community1_shortpath executable",
    )
    ap.add_argument(
        "--segmentation-onnx",
        type=Path,
        default=_REPO_ROOT / "cpp" / "artifacts" / "community1-segmentation.onnx",
        help="Segmentation ONNX for the C++ short path",
    )
    ap.add_argument(
        "--cpp-mode",
        choices=("oracle", "full"),
        default="oracle",
        help="oracle: C++ uses golden hard_clusters (default). full: VBx + ORT embedding, no oracle NPZ.",
    )
    ap.add_argument(
        "--cpp-embedding-onnx",
        type=Path,
        default=_REPO_ROOT / "cpp" / "artifacts" / "community1-embedding.onnx",
        help="Used when --cpp-mode full",
    )
    ap.add_argument(
        "--cpp-xvec",
        type=Path,
        default=None,
        help="xvec_transform.npz for full C++ mode (default: download from --checkpoint)",
    )
    ap.add_argument(
        "--cpp-plda",
        type=Path,
        default=None,
        help="plda.npz for full C++ mode (default: download from --checkpoint)",
    )
    ap.add_argument("--skip-cpp", action="store_true", help="Skip C++ binary; only Python DER + golden dump")
    ap.add_argument(
        "--skip-dump",
        action="store_true",
        help="Reuse wavs + golden under --work-dir (must already match indices and truncation)",
    )
    ap.add_argument(
        "--cpp-clustering-check",
        action="store_true",
        help=(
            "After C++ batch in --cpp-mode full, run clustering_golden_test on the first utterance only "
            "(requires cpp/build/clustering_golden_test and golden NPZ from the dump)."
        ),
    )
    args = ap.parse_args()

    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")

    if args.cpp_mode == "full":
        print(
            "Note (--cpp-mode full): the C++ column runs ORT embeddings + the ported VBx stack in C++; "
            "the Python column runs Torch embeddings + pyannote's clustering (SciPy/NumPy numerics). "
            "Those stages are not bit-identical, so ``hard_clusters`` and DER often differ a lot from "
            "Python even when segmentation and the eval WAV match the golden dump. "
            "Use ``--cpp-mode oracle`` to check end-to-end parity against the dumped pipeline "
            "(oracle ``hard_clusters_final.npz`` + same reconstruct path).",
            flush=True,
        )

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    from pyannote.audio import Pipeline
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    work: Path = args.work_dir.resolve()
    wav_dir = work / "wavs"
    golden_root = work / "golden"
    cpp_out = work / "cpp_out"
    cpp_out.mkdir(parents=True, exist_ok=True)

    token = _pick_token(args.token)
    ds = load_dataset("diarizers-community/callhome", args.subset, split=args.split)

    hi = args.start_index + args.num_files
    if args.start_index < 0 or hi > len(ds):
        raise SystemExit(
            f"row range [{args.start_index}, {hi}) out of range for split (len={len(ds)})"
        )

    stems: list[str] = []
    wav_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    durations: list[float] = []

    for i in range(args.start_index, hi):
        stem = _utterance_stem(args.subset, i, args.max_seconds)
        wav_path = wav_dir / f"{stem}.wav"
        row = ds[i]
        if not args.skip_dump:
            wav_dir.mkdir(parents=True, exist_ok=True)
            dur = _save_wav_row(row, wav_path, args.max_seconds)
            durations.append(dur)
        else:
            if not wav_path.is_file():
                raise SystemExit(f"--skip-dump but missing wav: {wav_path}")
            durations.append(_wav_duration_sec(wav_path))
        stems.append(stem)
        wav_paths.append(wav_path.resolve())
        rows.append(row)

    if not args.skip_dump:
        golden_root.mkdir(parents=True, exist_ok=True)
        dump_py = _REPO_ROOT / "cpp" / "scripts" / "dump_diarization_golden.py"
        cmd = [
            sys.executable,
            str(dump_py),
            "-o",
            str(golden_root),
            "--checkpoint",
            args.checkpoint,
            *[str(p) for p in wav_paths],
        ]
        if args.revision:
            cmd.extend(["--revision", args.revision])
        if token:
            cmd.extend(["--token", str(token)])
        print(
            f"Running golden dump: {dump_py.name} -o {golden_root} ({len(wav_paths)} wav(s))",
            flush=True,
        )
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)

    if not args.skip_cpp:
        cpp_bin = args.cpp_binary.resolve()
        if not cpp_bin.is_file():
            raise SystemExit(f"C++ binary not found: {cpp_bin} (build with scripts/build_cpp.sh)")
        onnx = args.segmentation_onnx.resolve()
        rf_json = golden_root / "receptive_field.json"
        snap_json = golden_root / "pipeline_snapshot.json"
        for p in (onnx, rf_json, snap_json):
            if not p.is_file():
                raise SystemExit(f"Missing required file for C++: {p}")

        list_file = work / "wav_list.txt"
        list_file.write_text("\n".join(str(p) for p in wav_paths) + "\n", encoding="utf-8")

        cmd_cpp = [
            str(cpp_bin),
            "--wav-list",
            str(list_file),
            "--artifact-base",
            str(golden_root),
            "--out-dir",
            str(cpp_out),
            "--segmentation-onnx",
            str(onnx),
            "--receptive-field",
            str(rf_json),
            "--pipeline-snapshot",
            str(snap_json),
        ]
        xvec_resolved: Path | None = None
        plda_resolved: Path | None = None
        if args.cpp_mode == "full":
            emb_onnx = args.cpp_embedding_onnx.resolve()
            if not emb_onnx.is_file():
                raise SystemExit(f"--cpp-mode full requires embedding ONNX: {emb_onnx}")
            if args.cpp_xvec is not None and args.cpp_plda is not None:
                xv = args.cpp_xvec.resolve()
                pl = args.cpp_plda.resolve()
            else:
                xv, pl = _hf_embedding_plda_paths(args.checkpoint, args.revision, token)
            xvec_resolved = xv
            plda_resolved = pl
            cmd_cpp.extend(
                [
                    "--embedding-onnx",
                    str(emb_onnx),
                    "--xvec-transform",
                    str(xv),
                    "--plda",
                    str(pl),
                ]
            )
        print("Running C++ batch:", cpp_bin.name, f"(cpp_mode={args.cpp_mode})", flush=True)
        subprocess.run(cmd_cpp, cwd=str(_REPO_ROOT), check=True)

        if (
            args.cpp_mode == "full"
            and args.cpp_clustering_check
            and stems
            and xvec_resolved is not None
            and plda_resolved is not None
        ):
            clu_bin = cpp_bin.parent / "clustering_golden_test"
            utter0 = golden_root / stems[0]
            if clu_bin.is_file() and utter0.is_dir():
                print(
                    f"Running clustering_golden_test on first utterance: {utter0.name}",
                    flush=True,
                )
                subprocess.run(
                    [str(clu_bin), str(utter0), str(xvec_resolved), str(plda_resolved)],
                    cwd=str(_REPO_ROOT),
                    check=False,
                )

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pipeline = Pipeline.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        token=token,
    )
    if pipeline is None:
        raise SystemExit("Pipeline.from_pretrained returned None (token / download error).")
    pipeline.to(device)

    py_ders: list[float] = []
    cpp_ders: list[float] = []

    print()
    print(f"{'idx':>4}  {'uri':<42}  {'DER py':>8}  {'DER cpp':>8}")
    print("-" * 72)

    for j, (i, stem, wav_path, row, dur) in enumerate(
        zip(
            range(args.start_index, hi),
            stems,
            wav_paths,
            rows,
            durations,
            strict=True,
        )
    ):
        uri = stem
        ref = _row_to_reference(row, crop_end=dur, uri=uri)
        uem = Timeline([Segment(0.0, dur)])

        file = {"audio": str(wav_path), "uri": uri}
        with torch.inference_mode():
            out = pipeline(file)
        hyp_py = out.speaker_diarization
        der_py = float(DiarizationErrorRate()(ref, hyp_py, uem=uem))
        py_ders.append(der_py)

        der_cpp = float("nan")
        if not args.skip_cpp:
            cpp_json = cpp_out / f"{stem}.json"
            if cpp_json.is_file():
                hyp_cpp = _json_diarization_to_annotation(cpp_json, uri=uri)
                der_cpp = float(DiarizationErrorRate()(ref, hyp_cpp, uem=uem))
        cpp_ders.append(der_cpp)

        print(f"{i:4d}  {uri:<42}  {100.0 * der_py:7.2f}%  {100.0 * der_cpp:7.2f}%")

    def _mean(xs: list[float]) -> float:
        vals = [x for x in xs if not np.isnan(x)]
        return float(np.mean(vals)) if vals else float("nan")

    print("-" * 72)
    print(
        f"{'mean':>4}  {'(over files with finite DER)':<42}  "
        f"{100.0 * _mean(py_ders):7.2f}%  {100.0 * _mean(cpp_ders):7.2f}%"
    )
    print()
    print("Reference: HF ``timestamps_*`` / ``speakers`` clipped to truncated WAV length.")
    print("Python: full SpeakerDiarization output on each WAV.")
    if args.cpp_mode == "oracle":
        print("C++: community1_shortpath with oracle hard_clusters from the golden dump (same checkpoint).")
    else:
        print(
            "C++: community1_shortpath full pipeline (segmentation + ORT embedding + VBx; no oracle clusters NPZ)."
        )


if __name__ == "__main__":
    main()
