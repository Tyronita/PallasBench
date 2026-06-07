#!/usr/bin/env python3
"""Standalone generation runner — sets all env vars internally."""
import os, sys

os.environ.setdefault("AZURE_API_ENDPOINT", "https://openai-shinka.cognitiveservices.azure.com/")
os.environ.setdefault("AZURE_API_VERSION", "2025-04-01-preview")
os.environ.setdefault("JAX_PALLAS_USE_MOSAIC_GPU", "1")

_NVIDIA = "/home/OLeary/.local/lib/python3.12/site-packages/nvidia"
import glob
_libs = ":".join(d for d in glob.glob(f"{_NVIDIA}/*/lib") if os.path.isdir(d))
os.environ["LD_LIBRARY_PATH"] = _libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PATH"] = f"{_NVIDIA}/cuda_nvcc/bin:" + os.environ.get("PATH", "")
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={_NVIDIA}/cuda_nvcc/nvvm/libdevice"

for p in ["/home/OLeary/jax", "/home/OLeary/Evolve", "."]:
    if p not in sys.path:
        sys.path.insert(0, p)

sys.argv = ["generate_archive.py", "run",
            "--n-variants", "20",
            "--model", "DeepSeek-V3-2",
            "--output", "results/archive_deepseek",
            "--resume"]

from scripts.generate_archive import cli
cli()
