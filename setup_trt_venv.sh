#!/usr/bin/env bash
set -euo pipefail

# Configurable venv directory via env var, defaults to .venv-trt
VENV_DIR=${VENV_DIR:-.venv-trt}

echo "[info] Creating virtual environment at: ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Numpy: uninstall then install specific version
python -m pip uninstall -y numpy || true
python -m pip install --no-cache-dir "numpy==2.1.2"

# Core libs
python -m pip install -U \
  "transformers>=4.43" \
  "accelerate>=0.33" \
  sentencepiece \
  onnx \
  "optimum>=1.21.4" \
  "onnxruntime-gpu>=1.18.0"

# TensorRT Python bindings (CUDA 12) + helpers (from NVIDIA index)
python -m pip install --extra-index-url https://pypi.nvidia.com \
  tensorrt-cu12 \
  polygraphy \
  onnx-graphsurgeon

cat <<'PY' | python
import sys
print("\n[env] Python:", sys.version)
try:
    import onnxruntime as ort
    print("[env] onnxruntime:", ort.__version__)
    print("[env] available providers:", ort.get_available_providers())
except Exception as e:
    print("[env] onnxruntime: ERROR:", e)
try:
    import tensorrt as trt
    print("[env] tensorrt:", trt.__version__)
except Exception as e:
    print("[env] tensorrt: NOT FOUND (", e, ")")
try:
    import torch
    from torch import cuda
    print("[env] torch:", torch.__version__, "cuda:", getattr(torch.version, 'cuda', None))
    print("[env] gpu:", cuda.get_device_name(0) if cuda.is_available() else "CPU")
except Exception as e:
    print("[env] torch: NOT INSTALLED (", e, ")")
PY

echo "\n[note] If TensorRT EP does not appear in providers, ensure system TensorRT libs are installed and LD_LIBRARY_PATH includes libnvinfer.*."
echo "[done] Environment ready. Activate with: source ${VENV_DIR}/bin/activate" 