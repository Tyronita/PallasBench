#!/usr/bin/env bash
# Full archive generation run — T4, 20 variants, DeepSeek-V3-2
# Usage: bash run_archive.sh
set -e

# -- Environment --
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:?need AZURE_OPENAI_API_KEY}"
export AZURE_API_ENDPOINT="https://openai-shinka.cognitiveservices.azure.com/"
export AZURE_API_VERSION="2025-04-01-preview"

# -- CUDA/JAX for T4 --
NVIDIA_PIP_BASE="/home/OLeary/.local/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS=$(find "$NVIDIA_PIP_BASE" -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
export PATH="/home/OLeary/.local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/OLeary/.local/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/libdevice"
export JAX_PALLAS_USE_MOSAIC_GPU=1
export PYTHONPATH="/home/OLeary/jax:/home/OLeary/Evolve:."

echo "[$(date)] Starting archive generation — 45 problems × 20 variants = ~900 rows"
python3 scripts/generate_archive.py run \
    --n-variants 20 \
    --model DeepSeek-V3-2 \
    --output results/archive_deepseek \
    --resume \
    2>&1 | tee results/archive_deepseek.log

echo "[$(date)] Done. Output in results/archive_deepseek/"
