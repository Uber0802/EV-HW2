#!/usr/bin/env bash
# Reproducible environment for EV-HW2 (conda + PyTorch + CUDA extensions).
#
# Prerequisites: conda, NVIDIA driver, git submodules initialized.
#
# Machine-specific knobs (set before running, or rely on defaults):
#   TORCH_CUDA_ARCH_LIST  GPU architectures for extension compile (see below).
#   CUDA_HOME             CUDA toolkit root; defaults to conda env prefix if
#                         cuda-toolkit is installed in the env (recommended).
#
# TORCH_CUDA_ARCH_LIST:
#   For faster builds, set only your GPU's compute capability, e.g.:
#     export TORCH_CUDA_ARCH_LIST=8.6    # RTX 30xx
#     export TORCH_CUDA_ARCH_LIST=8.9    # RTX 40xx
#     export TORCH_CUDA_ARCH_LIST=9.0         # Hopper
#     export TORCH_CUDA_ARCH_LIST=12.0        # Blackwell (requires CUDA 12.8+ nvcc)
#   Reference: https://developer.nvidia.com/cuda-gpus
#   Default below is a broad list so one checkout works on many lab machines
#   (compile will be slower).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-ev_hw2}"

# Broad default; override for a quicker, arch-specific build.
# Includes Blackwell (12.0). Submodule setup.py will sanitize this automatically
# when nvcc is older than 12.8.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;12.0}"

# --- 1. Create conda environment (optional; uncomment to bootstrap from scratch) ---
# echo "=== [1/5] Creating conda environment: $ENV_NAME (Python 3.10) ==="
# if conda env list | grep -q "^${ENV_NAME} "; then
#     echo "  Environment '$ENV_NAME' already exists — remove it first or use a different ENV_NAME."
#     exit 1
# fi
# conda create -n "$ENV_NAME" python=3.10 -y

# --- 2. CUDA toolkit in conda (optional; needed so nvcc matches PyTorch cu128 builds) ---
# echo ""
# echo "=== [2/5] Installing CUDA 12.8 toolkit into conda env ==="
# conda install -n "$ENV_NAME" -c nvidia/label/cuda-12.8.0 cuda-toolkit -y

# --- 3. Python dependencies (PyTorch + project packages) ---
echo ""
echo "=== [3/5] Installing Python packages from requirements.txt ==="
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt

# --- 4. Submodules (depth rasterizer + KNN) ---
echo ""
echo "=== [4/5] Checking git submodules ==="
if [ ! -f "submodules/depth-diff-gaussian-rasterization/setup.py" ]; then
    echo "  ERROR: submodules/depth-diff-gaussian-rasterization not found."
    echo "  Run:  git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "submodules/simple-knn/setup.py" ]; then
    echo "  ERROR: submodules/simple-knn not found."
    echo "  Run:  git submodule update --init --recursive"
    exit 1
fi

CONDA_PREFIX=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")
# Prefer explicit CUDA_HOME; else use conda prefix when nvcc is there.
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        CUDA_HOME="$CONDA_PREFIX"
    else
        echo "  WARNING: nvcc not found at \$CONDA_PREFIX/bin/nvcc and CUDA_HOME unset."
        echo "  Install cuda-toolkit in this conda env (see commented section 2) or set CUDA_HOME."
    fi
fi

echo ""
echo "=== [5/5] Building and installing CUDA extensions ==="
echo "  Using CUDA_HOME=${CUDA_HOME:-<unset>}  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "  Installing diff_gaussian_rasterization (depth-diff fork)..."
CUDA_HOME="${CUDA_HOME:-}" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    conda run -n "$ENV_NAME" pip install submodules/depth-diff-gaussian-rasterization --no-build-isolation

echo "  Installing simple-knn..."
CUDA_HOME="${CUDA_HOME:-}" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    conda run -n "$ENV_NAME" pip install submodules/simple-knn --no-build-isolation

# --- Verify ---
echo ""
echo "=== Verifying installation ==="
conda run -n "$ENV_NAME" python -c "
import torch
print(f'  torch                       : {torch.__version__}')
print(f'  CUDA available              : {torch.cuda.is_available()}')
import diff_gaussian_rasterization
print(f'  diff_gaussian_rasterization : OK')
import simple_knn
print(f'  simple_knn                  : OK')
import lpips
print(f'  lpips                       : OK')
"

echo ""
echo "Done. To use: conda activate $ENV_NAME"
