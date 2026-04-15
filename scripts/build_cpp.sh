#!/usr/bin/env bash
# Configure and build cpp/ (requires ONNXRUNTIME_ROOT).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${ONNXRUNTIME_ROOT:-}" ]] && [[ -d /opt/homebrew/opt/onnxruntime ]]; then
  export ONNXRUNTIME_ROOT="/opt/homebrew/opt/onnxruntime"
fi
if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "error: set ONNXRUNTIME_ROOT to your onnxruntime prebuilt tree (include/ + lib/)." >&2
  echo "  See cpp/README.md" >&2
  exit 1
fi

cmake -S cpp -B cpp/build "-DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
cmake --build cpp/build --parallel

echo "Built:"
echo "  ${ROOT}/cpp/build/segmentation_golden_test"
echo "  ${ROOT}/cpp/build/embedding_golden_test"
echo "  ${ROOT}/cpp/build/speaker_count_golden_test"
echo "  ${ROOT}/cpp/build/reconstruct_golden_test"
echo "  ${ROOT}/cpp/build/annotation_golden_test"
echo "  ${ROOT}/cpp/build/cpp-annote-cli"
echo "  ${ROOT}/cpp/build/full_segmentation_window_parity_test"
echo "  ${ROOT}/cpp/build/clustering_golden_test"
echo "  ${ROOT}/cpp/build/frozen_golden_inputs_test"
echo "  ${ROOT}/cpp/build/vbx_parity_test"
echo "  ${ROOT}/cpp/build/linkage_fcluster_golden_test"
echo "  ${ROOT}/cpp/build/knf_smoke_test"
