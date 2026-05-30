from __future__ import annotations

from typing import Any

import jax


def capture_jaxpr(fn, *sample_args, **sample_kwargs) -> str:
    try:
        jaxpr = jax.make_jaxpr(fn)(*sample_args, **sample_kwargs)
        return str(jaxpr)
    except Exception as e:
        return f"<jaxpr capture failed: {e}>"


def capture_stablehlo(fn, *sample_args, **sample_kwargs) -> str:
    try:
        lowered = jax.jit(fn).lower(*sample_args, **sample_kwargs)
        return lowered.as_text()
    except Exception as e:
        return f"<stablehlo capture failed: {e}>"


def capture_triton_mlir(fn, *sample_args, **sample_kwargs) -> str | None:
    try:
        import os

        os.environ.setdefault("JAX_DUMP_IR_IN_LLVM_IR_FORMAT", "true")
        lowered = jax.jit(fn).lower(*sample_args, **sample_kwargs)
        triton_ir = getattr(lowered, "triton_ir", None)
        if triton_ir is not None:
            return str(triton_ir)
        return "<no triton_ir attribute on lowered object>"
    except Exception as e:
        return f"<triton mlir capture failed: {e}>"


def capture_all_ir(
    fn, *sample_args, **sample_kwargs
) -> dict[str, str | None]:
    return {
        "jaxpr": capture_jaxpr(fn, *sample_args, **sample_kwargs),
        "stablehlo": capture_stablehlo(fn, *sample_args, **sample_kwargs),
        "triton_mlir": capture_triton_mlir(fn, *sample_args, **sample_kwargs),
    }
