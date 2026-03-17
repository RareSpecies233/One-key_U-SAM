#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ORT_PREFIX="${ORT_PREFIX:-/opt/homebrew/opt/onnxruntime}"
ORT_INCLUDE_DIR="${ORT_INCLUDE_DIR:-$ORT_PREFIX/include}"
ORT_LIB_DIR="${ORT_LIB_DIR:-$ORT_PREFIX/lib}"
OUTPUT_BIN="${OUTPUT_BIN:-$SCRIPT_DIR/infer_rectum_onnx}"

if [[ ! -f "$ORT_INCLUDE_DIR/onnxruntime/onnxruntime_cxx_api.h" ]]; then
  echo "ONNX Runtime headers not found under $ORT_INCLUDE_DIR" >&2
  echo "Install with: brew install onnxruntime" >&2
  exit 1
fi

if [[ ! -f "$ORT_LIB_DIR/libonnxruntime.dylib" ]]; then
  echo "ONNX Runtime library not found under $ORT_LIB_DIR" >&2
  echo "Install with: brew install onnxruntime" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_BIN")"

clang++ \
  -std=c++17 \
  -O3 \
  -Wall \
  -Wextra \
  -I"$ORT_INCLUDE_DIR" \
  -I"$REPO_ROOT/cppInfer" \
  -I"$REPO_ROOT/cppInfer/cnpy" \
  "$REPO_ROOT/cppInfer/infer_rectum_onnx.cpp" \
  "$REPO_ROOT/cppInfer/cnpy/cnpy.cpp" \
  -L"$ORT_LIB_DIR" \
  -Wl,-rpath,"$ORT_LIB_DIR" \
  -lonnxruntime \
  -lz \
  -o "$OUTPUT_BIN"

echo "Built $OUTPUT_BIN"