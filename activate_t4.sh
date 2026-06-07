#!/usr/bin/env bash
# Source this before running any JAX/Pallas commands on this machine:
#   source activate_t4.sh
#
# GPU STATUS (T4 / sm_75):
#   ✓ JAX XLA ops run on GPU (matmul, elementwise, etc.)
#   ✓ Pallas CORRECTNESS checks via interpret=True (CPU emulation)
#   ✗ Pallas PERFORMANCE on T4: JAX 0.10.1 Pallas requires sm_80+
#       - Mosaic GPU backend: uses cp.async.bulk (Ampere+)
#       - Triton backend: hard-coded sm_80+ minimum
#   → For real Pallas performance, run on sm_80+ (A100/RTX30xx/RTX40xx)
#   → The archive pipeline generates code+IR+correctness correctly on T4
#
# AZURE LLM:
#   export AZURE_OPENAI_API_KEY=<your-key>    (run: az account get-access-token)
#   Models: DeepSeek-V3-2, Kimi-K2-6, gpt-5-codex, gpt-4o-mini

export NVIDIA_PIP_BASE="/home/OLeary/.local/lib/python3.12/site-packages/nvidia"
export NVIDIA_LIBS=$(find $NVIDIA_PIP_BASE -name "lib" -type d 2>/dev/null | tr '\n' ':')
export PTXAS_PATH="$NVIDIA_PIP_BASE/cuda_nvcc/bin"
export LIBDEVICE_PATH="$NVIDIA_PIP_BASE/cuda_nvcc/nvvm/libdevice"

export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
export PATH="${PTXAS_PATH}:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${LIBDEVICE_PATH}"
export JAX_PALLAS_USE_MOSAIC_GPU=1
export PYTHONPATH="/home/OLeary/jax:/home/OLeary/Evolve:${PYTHONPATH:-}"

echo "T4 JAX environment activated."
echo "  ptxas:     $(ptxas --version 2>&1 | head -1)"
echo "  Python:    $(python3 --version)"
python3 -c "
import jax
devs = jax.devices()
print('  JAX:      ', jax.__version__)
print('  Devices:  ', devs)
gpu = any(d.platform == 'gpu' for d in devs)
if gpu:
    print('  Status:    GPU OK for JAX ops | Pallas via interpret=True (sm_75)')
else:
    print('  Status:    CPU only — check LD_LIBRARY_PATH')
" 2>/dev/null
