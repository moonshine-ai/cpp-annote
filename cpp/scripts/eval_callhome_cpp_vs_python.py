#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026- pyannote.audio contributors
#
# Evaluate Diarization Error Rate (DER) on the first n clips from Hugging Face
# ``diarizers-community/callhome`` (English subset by default): reference labels
# from dataset timestamps vs (1) full ``SpeakerDiarization`` in Python and
# (2) C++ ``community1_shortpath`` using oracle clusters from a golden dump of
# the same pipeline (same configuration as ``dump_diarization_golden.py``).

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
            "(oracle clusters from golden dump)."
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
    ap.add_argument("--skip-cpp", action="store_true", help="Skip C++ binary; only Python DER + golden dump")
    ap.add_argument(
        "--skip-dump",
        action="store_true",
        help="Reuse wavs + golden under --work-dir (must already match indices and truncation)",
    )
    args = ap.parse_args()

    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")

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
        if token:
            pass
        print("Running C++ batch:", cpp_bin.name, flush=True)
        subprocess.run(cmd_cpp, cwd=str(_REPO_ROOT), check=True)

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
    print("C++: community1_shortpath with oracle hard_clusters from the golden dump (same checkpoint).")


if __name__ == "__main__":
    main()
