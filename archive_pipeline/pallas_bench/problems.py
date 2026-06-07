"""
All 45 PallasBench problem specs: JAX reference code, seed Pallas kernel,
input shape generators, and metadata.

Layout: 27 L1 (single ops) + 13 L2 (fused patterns) + 5 L3 (architecture).
Seed Pallas kernels are the GPU-fixed versions from pallasbench-robust
(block size clamped for T4's 64 KB shared memory / Triton 1M element limit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import json
import numpy as np

# ---------------------------------------------------------------------------
# Problem descriptor
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    task_id: int
    name: str           # e.g. "relu"
    level: int          # 1 / 2 / 3
    category: str       # e.g. "activation"
    github_url: str
    github_raw_url: str
    jax_module: str     # JAX reference as a class with __call__
    jax_functional: str # JAX reference as a plain function
    seed_pallas: str    # GPU-fixed Pallas kernel (from pallasbench-robust)
    original_pallas: str = ""  # original TPU-oriented source (may differ)
    input_shapes: list[tuple] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)

    def input_shapes_json(self) -> str:
        return json.dumps(self.input_shapes)

    def input_dtypes_json(self) -> str:
        return json.dumps(self.input_dtypes)


_BASE_URL = "https://github.com/Tyronita/PallasBench/blob/main"
_RAW_URL  = "https://raw.githubusercontent.com/Tyronita/PallasBench/main"

def _urls(path: str) -> dict:
    return {
        "github_url":     f"{_BASE_URL}/{path}",
        "github_raw_url": f"{_RAW_URL}/{path}",
    }


# ---------------------------------------------------------------------------
# Helper: T4-safe block sizes
# T4 (sm_75): 64 KB shared memory, Triton block element cap ~1M
# ---------------------------------------------------------------------------
_BM = 128  # safe row block
_BN = 128  # safe col block
_BK = 64   # inner reduction block


# ===========================================================================
# LEVEL 1 — Single Operators (27 problems)
# ===========================================================================

# ── Activations (5) ─────────────────────────────────────────────────────────

_p1 = Problem(
    task_id=1, name="relu", level=1, category="activation",
    **_urls("kernels/level1/relu.py"),
    input_shapes=[(4096,)], input_dtypes=["float32"],
    jax_functional="""
def relu(x):
    import jax.numpy as jnp
    return jnp.maximum(x, 0.0)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.maximum(x, 0.0)
""",
    seed_pallas="""
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def relu_kernel(x_ref, o_ref):
    o_ref[...] = jnp.maximum(x_ref[...], 0.0)

def relu(x):
    n = x.shape[0]
    block = min(1024, n)
    return pl.pallas_call(
        relu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p2 = Problem(
    task_id=2, name="gelu", level=1, category="activation",
    **_urls("kernels/level1/gelu.py"),
    input_shapes=[(4096,)], input_dtypes=["float32"],
    jax_functional="""
def gelu(x):
    import jax.numpy as jnp
    c = 0.7978845608028654
    return 0.5 * x * (1.0 + jnp.tanh(c * (x + 0.044715 * x**3)))
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.nn as jnn
        return jnn.gelu(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl, math

def gelu_kernel(x_ref, o_ref):
    x = x_ref[...]
    c = 0.7978845608028654
    o_ref[...] = 0.5 * x * (1.0 + jnp.tanh(c * (x + 0.044715 * x**3)))

def gelu(x):
    n = x.shape[0]; block = min(1024, n)
    return pl.pallas_call(
        gelu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p3 = Problem(
    task_id=3, name="silu", level=1, category="activation",
    **_urls("kernels/level1/silu.py"),
    input_shapes=[(4096,)], input_dtypes=["float32"],
    jax_functional="""
def silu(x):
    import jax, jax.numpy as jnp
    return x * jax.nn.sigmoid(x)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.nn as jnn
        return jnn.silu(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def silu_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = x * jax.nn.sigmoid(x)

def silu(x):
    n = x.shape[0]; block = min(1024, n)
    return pl.pallas_call(
        silu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p4 = Problem(
    task_id=4, name="sigmoid", level=1, category="activation",
    **_urls("kernels/level1/sigmoid.py"),
    input_shapes=[(4096,)], input_dtypes=["float32"],
    jax_functional="""
def sigmoid(x):
    import jax.numpy as jnp
    return 1.0 / (1.0 + jnp.exp(-x))
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.nn as jnn
        return jnn.sigmoid(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def sigmoid_kernel(x_ref, o_ref):
    o_ref[...] = 1.0 / (1.0 + jnp.exp(-x_ref[...]))

def sigmoid(x):
    n = x.shape[0]; block = min(1024, n)
    return pl.pallas_call(
        sigmoid_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p5 = Problem(
    task_id=5, name="tanh", level=1, category="activation",
    **_urls("kernels/level1/tanh.py"),
    input_shapes=[(4096,)], input_dtypes=["float32"],
    jax_functional="""
def tanh(x):
    import jax.numpy as jnp
    return jnp.tanh(x)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.tanh(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def tanh_kernel(x_ref, o_ref):
    o_ref[...] = jnp.tanh(x_ref[...])

def tanh(x):
    n = x.shape[0]; block = min(1024, n)
    return pl.pallas_call(
        tanh_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

# ── Normalization (2) ────────────────────────────────────────────────────────

_p6 = Problem(
    task_id=6, name="layernorm", level=1, category="normalization",
    **_urls("kernels/level1/layernorm.py"),
    input_shapes=[(512, 1024), (1024,), (1024,)], input_dtypes=["float32", "float32", "float32"],
    jax_functional="""
def layernorm(x, weight, bias, eps=1e-5):
    import jax.numpy as jnp
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return weight * (x - mean) / jnp.sqrt(var + eps) + bias
""",
    jax_module="""
class Model:
    def __call__(self, x, weight, bias):
        import jax.numpy as jnp
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x,  axis=-1, keepdims=True)
        return weight * (x - mean) / jnp.sqrt(var + 1e-5) + bias
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def layernorm_kernel(x_ref, w_ref, b_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    o_ref[...] = (w_ref[...] * (x - mean) / jnp.sqrt(var + 1e-5) + b_ref[...]).astype(x_ref.dtype)

def layernorm(x, weight, bias):
    B, D = x.shape; bm = min(16, B); bn = min(D, 1024)
    return pl.pallas_call(
        layernorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0)),
                  pl.BlockSpec((bn,),    lambda i: (0,)),
                  pl.BlockSpec((bn,),    lambda i: (0,))],
        out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
    )(x, weight, bias)
""",
)

_p7 = Problem(
    task_id=7, name="rmsnorm", level=1, category="normalization",
    **_urls("kernels/level1/rmsnorm.py"),
    input_shapes=[(512, 1024), (1024,)], input_dtypes=["float32", "float32"],
    jax_functional="""
def rmsnorm(x, weight, eps=1e-6):
    import jax.numpy as jnp
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return weight * x / rms
""",
    jax_module="""
class Model:
    def __call__(self, x, weight):
        import jax.numpy as jnp
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return weight * x / rms
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def rmsnorm_kernel(x_ref, w_ref, o_ref):
    x = x_ref[...].astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    o_ref[...] = (w_ref[...] * x / rms).astype(x_ref.dtype)

def rmsnorm(x, weight):
    B, D = x.shape; bm = min(16, B); bn = min(D, 1024)
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0)),
                  pl.BlockSpec((bn,),    lambda i: (0,))],
        out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
    )(x, weight)
""",
)

# ── Linear algebra (3) ───────────────────────────────────────────────────────

_p8 = Problem(
    task_id=8, name="matmul", level=1, category="matmul",
    **_urls("kernels/level1/matmul.py"),
    input_shapes=[(512, 512), (512, 512)], input_dtypes=["float32", "float32"],
    jax_functional="""
def matmul(a, b):
    import jax.numpy as jnp
    return a @ b
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax.numpy as jnp
        return jnp.matmul(a, b)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
BM, BN, BK = 128, 128, 64

def matmul_kernel(a_ref, b_ref, o_ref):
    # K is static: each block covers the full K dimension
    acc = jnp.zeros((BM, BN), dtype=jnp.float32)
    acc = acc + a_ref[...].astype(jnp.float32) @ b_ref[...].astype(jnp.float32)
    o_ref[...] = acc.astype(o_ref.dtype)

def matmul(a, b):
    M, K = a.shape; _, N = b.shape
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                  pl.BlockSpec((K, BN), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
""",
)

_p9 = Problem(
    task_id=9, name="batched_matmul", level=1, category="matmul",
    **_urls("kernels/level1/batched_matmul.py"),
    input_shapes=[(8, 128, 128), (8, 128, 128)], input_dtypes=["float32", "float32"],
    jax_functional="""
def batched_matmul(a, b):
    import jax.numpy as jnp
    return jnp.einsum('bij,bjk->bik', a, b)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax
        return jax.vmap(lambda x, y: x @ y)(a, b)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def batched_matmul(a, b):
    return jax.vmap(lambda x, y: x @ y)(a, b)
""",
)

_p10 = Problem(
    task_id=10, name="outer_product", level=1, category="matmul",
    **_urls("kernels/level1/outer_product.py"),
    input_shapes=[(1024,), (1024,)], input_dtypes=["float32", "float32"],
    jax_functional="""
def outer_product(a, b):
    import jax.numpy as jnp
    return jnp.outer(a, b)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax.numpy as jnp
        return jnp.outer(a, b)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
BM, BN = 128, 128

def outer_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = jnp.outer(a_ref[...], b_ref[...])

def outer_product(a, b):
    M, N = a.shape[0], b.shape[0]
    return pl.pallas_call(
        outer_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM,), lambda i, j: (i,)),
                  pl.BlockSpec((BN,), lambda i, j: (j,))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
""",
)

# ── Reductions (3) ───────────────────────────────────────────────────────────

_p11 = Problem(
    task_id=11, name="reduce_sum", level=1, category="reduce",
    **_urls("kernels/level1/reduce_sum.py"),
    input_shapes=[(4096, 512)], input_dtypes=["float32"],
    jax_functional="""
def reduce_sum(x):
    import jax.numpy as jnp
    return jnp.sum(x, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.sum(x, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def reduce_sum_kernel(x_ref, o_ref):
    o_ref[...] = jnp.sum(x_ref[...], axis=-1)

def reduce_sum(x):
    B, D = x.shape; bm = min(128, B); bn = min(512, D)
    return pl.pallas_call(
        reduce_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((B,), x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm,),    lambda i: (i,)),
    )(x)
""",
)

_p12 = Problem(
    task_id=12, name="reduce_max", level=1, category="reduce",
    **_urls("kernels/level1/reduce_max.py"),
    input_shapes=[(4096, 512)], input_dtypes=["float32"],
    jax_functional="""
def reduce_max(x):
    import jax.numpy as jnp
    return jnp.max(x, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.max(x, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def reduce_max_kernel(x_ref, o_ref):
    o_ref[...] = jnp.max(x_ref[...], axis=-1)

def reduce_max(x):
    B, D = x.shape; bm = min(128, B); bn = min(512, D)
    return pl.pallas_call(
        reduce_max_kernel,
        out_shape=jax.ShapeDtypeStruct((B,), x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm,),    lambda i: (i,)),
    )(x)
""",
)

_p13 = Problem(
    task_id=13, name="reduce_mean", level=1, category="reduce",
    **_urls("kernels/level1/reduce_mean.py"),
    input_shapes=[(4096, 512)], input_dtypes=["float32"],
    jax_functional="""
def reduce_mean(x):
    import jax.numpy as jnp
    return jnp.mean(x, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.mean(x, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def reduce_mean_kernel(x_ref, o_ref):
    o_ref[...] = jnp.mean(x_ref[...], axis=-1)

def reduce_mean(x):
    B, D = x.shape; bm = min(128, B); bn = min(512, D)
    return pl.pallas_call(
        reduce_mean_kernel,
        out_shape=jax.ShapeDtypeStruct((B,), x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm,),    lambda i: (i,)),
    )(x)
""",
)

# ── Softmax (2) ──────────────────────────────────────────────────────────────

_p14 = Problem(
    task_id=14, name="softmax", level=1, category="softmax",
    **_urls("kernels/level1/softmax.py"),
    input_shapes=[(512, 1024)], input_dtypes=["float32"],
    jax_functional="""
def softmax(x):
    import jax.nn as jnn
    return jnn.softmax(x, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.nn as jnn
        return jnn.softmax(x, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    x_max = jnp.max(x, axis=-1, keepdims=True)
    e = jnp.exp(x - x_max)
    o_ref[...] = e / jnp.sum(e, axis=-1, keepdims=True)

def softmax(x):
    B, D = x.shape; bm = min(16, B); bn = min(D, 1024)
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
    )(x)
""",
)

_p15 = Problem(
    task_id=15, name="log_softmax", level=1, category="softmax",
    **_urls("kernels/level1/log_softmax.py"),
    input_shapes=[(512, 1024)], input_dtypes=["float32"],
    jax_functional="""
def log_softmax(x):
    import jax.nn as jnn
    return jnn.log_softmax(x, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.nn as jnn
        return jnn.log_softmax(x, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def log_softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    x_max = jnp.max(x, axis=-1, keepdims=True)
    shifted = x - x_max
    o_ref[...] = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

def log_softmax(x):
    B, D = x.shape; bm = min(16, B); bn = min(D, 1024)
    return pl.pallas_call(
        log_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
    )(x)
""",
)

# ── Transcendentals (3) ──────────────────────────────────────────────────────

_p16 = Problem(
    task_id=16, name="exp", level=1, category="elementwise",
    **_urls("kernels/level1/exp.py"),
    input_shapes=[(8192,)], input_dtypes=["float32"],
    jax_functional="""
def exp(x):
    import jax.numpy as jnp
    return jnp.exp(x)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.exp(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def exp_kernel(x_ref, o_ref):
    o_ref[...] = jnp.exp(x_ref[...])

def exp(x):
    n = x.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        exp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p17 = Problem(
    task_id=17, name="log", level=1, category="elementwise",
    **_urls("kernels/level1/log.py"),
    input_shapes=[(8192,)], input_dtypes=["float32"],
    jax_functional="""
def log(x):
    import jax.numpy as jnp
    return jnp.log(x)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.log(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def log_kernel(x_ref, o_ref):
    o_ref[...] = jnp.log(x_ref[...])

def log(x):
    n = x.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        log_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

_p18 = Problem(
    task_id=18, name="rsqrt", level=1, category="elementwise",
    **_urls("kernels/level1/rsqrt.py"),
    input_shapes=[(8192,)], input_dtypes=["float32"],
    jax_functional="""
def rsqrt(x):
    import jax.numpy as jnp
    return jnp.reciprocal(jnp.sqrt(x))
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.lax as lax
        return lax.rsqrt(x)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def rsqrt_kernel(x_ref, o_ref):
    o_ref[...] = jax.lax.rsqrt(x_ref[...])

def rsqrt(x):
    n = x.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        rsqrt_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

# ── Elementwise (3) ──────────────────────────────────────────────────────────

_p19 = Problem(
    task_id=19, name="add", level=1, category="elementwise",
    **_urls("kernels/level1/add.py"),
    input_shapes=[(8192,), (8192,)], input_dtypes=["float32", "float32"],
    jax_functional="""
def add(a, b):
    return a + b
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        return a + b
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def add_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = a_ref[...] + b_ref[...]

def add(a, b):
    n = a.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,)),
                  pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(a, b)
""",
)

_p20 = Problem(
    task_id=20, name="multiply", level=1, category="elementwise",
    **_urls("kernels/level1/multiply.py"),
    input_shapes=[(8192,), (8192,)], input_dtypes=["float32", "float32"],
    jax_functional="""
def multiply(a, b):
    return a * b
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        return a * b
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def mul_kernel(a_ref, b_ref, o_ref):
    o_ref[...] = a_ref[...] * b_ref[...]

def multiply(a, b):
    n = a.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        mul_kernel,
        out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,)),
                  pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(a, b)
""",
)

_p21 = Problem(
    task_id=21, name="clamp", level=1, category="elementwise",
    **_urls("kernels/level1/clamp.py"),
    input_shapes=[(8192,)], input_dtypes=["float32"],
    jax_functional="""
def clamp(x, lo=-1.0, hi=1.0):
    import jax.numpy as jnp
    return jnp.clip(x, lo, hi)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        return jnp.clip(x, -1.0, 1.0)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def clamp_kernel(x_ref, o_ref):
    o_ref[...] = jnp.clip(x_ref[...], -1.0, 1.0)

def clamp(x):
    n = x.shape[0]; block = min(2048, n)
    return pl.pallas_call(
        clamp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x)
""",
)

# ── Loss functions (3) ───────────────────────────────────────────────────────

_p22 = Problem(
    task_id=22, name="cross_entropy", level=1, category="loss",
    **_urls("kernels/level1/cross_entropy.py"),
    input_shapes=[(512, 1024), (512,)], input_dtypes=["float32", "int32"],
    jax_functional="""
def cross_entropy(logits, labels):
    import jax, jax.numpy as jnp
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(labels.shape[0]), labels]
""",
    jax_module="""
class Model:
    def __call__(self, logits, labels):
        import jax, jax.numpy as jnp
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -log_probs[jnp.arange(labels.shape[0]), labels]
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -log_probs[jnp.arange(labels.shape[0]), labels]
""",
)

_p23 = Problem(
    task_id=23, name="mse_loss", level=1, category="loss",
    **_urls("kernels/level1/mse_loss.py"),
    input_shapes=[(4096,), (4096,)], input_dtypes=["float32", "float32"],
    jax_functional="""
def mse_loss(pred, target):
    import jax.numpy as jnp
    return jnp.mean((pred - target) ** 2)
""",
    jax_module="""
class Model:
    def __call__(self, pred, target):
        import jax.numpy as jnp
        return jnp.mean((pred - target) ** 2)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def mse_kernel(pred_ref, tgt_ref, o_ref):
    diff = pred_ref[...] - tgt_ref[...]
    o_ref[...] = diff * diff

def mse_loss(pred, target):
    n = pred.shape[0]; block = min(2048, n)
    sq = pl.pallas_call(
        mse_kernel,
        out_shape=jax.ShapeDtypeStruct(pred.shape, pred.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,)),
                  pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(pred, target)
    return jnp.mean(sq)
""",
)

_p24 = Problem(
    task_id=24, name="cosine_sim", level=1, category="loss",
    **_urls("kernels/level1/cosine_sim.py"),
    input_shapes=[(512, 256), (512, 256)], input_dtypes=["float32", "float32"],
    jax_functional="""
def cosine_sim(a, b):
    import jax.numpy as jnp
    a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return jnp.sum(a_norm * b_norm, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax.numpy as jnp
        a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
        return jnp.sum(a_norm * b_norm, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def cosine_sim(a, b):
    a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return jnp.sum(a_norm * b_norm, axis=-1)
""",
)

# ── Data manipulation (3) ────────────────────────────────────────────────────

_p25 = Problem(
    task_id=25, name="embedding_lookup", level=1, category="index",
    **_urls("kernels/level1/embedding_lookup.py"),
    input_shapes=[(8192, 64), (512,)], input_dtypes=["float32", "int32"],
    jax_functional="""
def embedding_lookup(table, indices):
    return table[indices]
""",
    jax_module="""
class Model:
    def __call__(self, table, indices):
        return table[indices]
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: embedding_lookup with integer indexing cannot be implemented
# directly in Pallas/Triton (non-array ops). Falls back to JAX gather.
def embedding_lookup(table, indices):
    return table[indices]
""",
)

_p26 = Problem(
    task_id=26, name="one_hot", level=1, category="index",
    **_urls("kernels/level1/one_hot.py"),
    input_shapes=[(512,)], input_dtypes=["int32"],
    jax_functional="""
def one_hot(indices):
    import jax
    return jax.nn.one_hot(indices, 1024)
""",
    jax_module="""
class Model:
    def __call__(self, indices, num_classes=1024):
        import jax.nn as jnn
        return jnn.one_hot(indices, num_classes)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

def one_hot(indices):
    return jax.nn.one_hot(indices, 1024)
""",
)

_p27 = Problem(
    task_id=27, name="nucleotide_onehot", level=1, category="index",
    **_urls("kernels/level1/nucleotide_onehot.py"),
    input_shapes=[(4096,)], input_dtypes=["int32"],
    jax_functional="""
def nucleotide_onehot(seq):
    import jax, jax.numpy as jnp
    return jax.nn.one_hot(seq, 4)
""",
    jax_module="""
class Model:
    def __call__(self, seq):
        import jax
        return jax.nn.one_hot(seq, 4)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def nucleotide_onehot_kernel(seq_ref, o_ref):
    idx = seq_ref[...]
    o_ref[...] = jax.nn.one_hot(idx, 4)

def nucleotide_onehot(seq):
    n = seq.shape[0]; block = min(1024, n)
    return pl.pallas_call(
        nucleotide_onehot_kernel,
        out_shape=jax.ShapeDtypeStruct((n, 4), jnp.float32),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block, 4), lambda i: (i, 0)),
    )(seq)
""",
)

# ===========================================================================
# LEVEL 2 — Fusion Patterns (13 problems, task_ids 28–40)
# ===========================================================================

_p28 = Problem(
    task_id=28, name="matmul_relu", level=2, category="fused",
    **_urls("kernels/level2/matmul_relu.py"),
    input_shapes=[(512, 512), (512, 512)], input_dtypes=["float32", "float32"],
    jax_functional="""
def matmul_relu(a, b):
    import jax.numpy as jnp
    return jnp.maximum(a @ b, 0.0)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax.numpy as jnp
        return jnp.maximum(a @ b, 0.0)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
BM, BN, BK = 128, 128, 64

def matmul_relu_kernel(a_ref, b_ref, o_ref):
    acc = a_ref[...].astype(jnp.float32) @ b_ref[...].astype(jnp.float32)
    o_ref[...] = jnp.maximum(acc, 0.0).astype(o_ref.dtype)

def matmul_relu(a, b):
    M, K = a.shape; _, N = b.shape
    return pl.pallas_call(
        matmul_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                  pl.BlockSpec((K, BN), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
""",
)

_p29 = Problem(
    task_id=29, name="matmul_gelu", level=2, category="fused",
    **_urls("kernels/level2/matmul_gelu.py"),
    input_shapes=[(512, 512), (512, 512)], input_dtypes=["float32", "float32"],
    jax_functional="""
def matmul_gelu(a, b):
    import jax, jax.numpy as jnp
    return jax.nn.gelu(a @ b)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax
        return jax.nn.gelu(a @ b)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
BM, BN, BK = 128, 128, 64

def matmul_gelu_kernel(a_ref, b_ref, o_ref):
    acc = a_ref[...].astype(jnp.float32) @ b_ref[...].astype(jnp.float32)
    c = 0.7978845608028654
    o_ref[...] = (0.5 * acc * (1.0 + jnp.tanh(c * (acc + 0.044715 * acc**3)))).astype(o_ref.dtype)

def matmul_gelu(a, b):
    M, K = a.shape; _, N = b.shape
    return pl.pallas_call(
        matmul_gelu_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                  pl.BlockSpec((K, BN), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
""",
)

_p30 = Problem(
    task_id=30, name="matmul_silu", level=2, category="fused",
    **_urls("kernels/level2/matmul_silu.py"),
    input_shapes=[(512, 512), (512, 512)], input_dtypes=["float32", "float32"],
    jax_functional="""
def matmul_silu(a, b):
    import jax
    out = a @ b
    return out * jax.nn.sigmoid(out)
""",
    jax_module="""
class Model:
    def __call__(self, a, b):
        import jax
        out = a @ b
        return out * jax.nn.sigmoid(out)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
BM, BN, BK = 128, 128, 64

def matmul_silu_kernel(a_ref, b_ref, o_ref):
    acc = a_ref[...].astype(jnp.float32) @ b_ref[...].astype(jnp.float32)
    o_ref[...] = (acc * jax.nn.sigmoid(acc)).astype(o_ref.dtype)

def matmul_silu(a, b):
    M, K = a.shape; _, N = b.shape
    return pl.pallas_call(
        matmul_silu_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), a.dtype),
        grid=(M // BM, N // BN),
        in_specs=[pl.BlockSpec((BM, K), lambda i, j: (i, 0)),
                  pl.BlockSpec((K, BN), lambda i, j: (0, j))],
        out_specs=pl.BlockSpec((BM, BN), lambda i, j: (i, j)),
    )(a, b)
""",
)

_p31 = Problem(
    task_id=31, name="linear_bias_relu", level=2, category="fused",
    **_urls("kernels/level2/linear_bias_relu.py"),
    input_shapes=[(512, 512), (512, 512), (512,)], input_dtypes=["float32"]*3,
    jax_functional="""
def linear_bias_relu(x, w, b):
    import jax.numpy as jnp
    return jnp.maximum(x @ w + b, 0.0)
""",
    jax_module="""
class Model:
    def __call__(self, x, w, b):
        import jax.numpy as jnp
        return jnp.maximum(x @ w + b, 0.0)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def linear_bias_relu(x, w, b):
    return jnp.maximum(x @ w + b, 0.0)
""",
)

_p32 = Problem(
    task_id=32, name="rmsnorm_residual", level=2, category="fused",
    **_urls("kernels/level2/rmsnorm_residual.py"),
    input_shapes=[(512, 1024), (512, 1024), (1024,)], input_dtypes=["float32"]*3,
    jax_functional="""
def rmsnorm_residual(x, residual, weight, eps=1e-6):
    import jax.numpy as jnp
    x = x + residual
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return weight * x / rms
""",
    jax_module="""
class Model:
    def __call__(self, x, residual, weight):
        import jax.numpy as jnp
        x = x + residual
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        return weight * x / rms
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def rmsnorm_residual_kernel(x_ref, r_ref, w_ref, o_ref):
    x = (x_ref[...] + r_ref[...]).astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    o_ref[...] = (w_ref[...] * x / rms).astype(x_ref.dtype)

def rmsnorm_residual(x, residual, weight):
    B, D = x.shape; bm = min(16, B); bn = min(D, 1024)
    return pl.pallas_call(
        rmsnorm_residual_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, bn), lambda i: (i, 0)),
                  pl.BlockSpec((bm, bn), lambda i: (i, 0)),
                  pl.BlockSpec((bn,),    lambda i: (0,))],
        out_specs=pl.BlockSpec((bm, bn), lambda i: (i, 0)),
    )(x, residual, weight)
""",
)

_p33 = Problem(
    task_id=33, name="layernorm_residual", level=2, category="fused",
    **_urls("kernels/level2/layernorm_residual.py"),
    input_shapes=[(512, 1024), (512, 1024), (1024,), (1024,)], input_dtypes=["float32"]*4,
    jax_functional="""
def layernorm_residual(x, residual, weight, bias, eps=1e-5):
    import jax.numpy as jnp
    x = x + residual
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return weight * (x - mean) / jnp.sqrt(var + eps) + bias
""",
    jax_module="""
class Model:
    def __call__(self, x, residual, weight, bias):
        import jax.numpy as jnp
        x = x + residual
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var  = jnp.var(x,  axis=-1, keepdims=True)
        return weight * (x - mean) / jnp.sqrt(var + 1e-5) + bias
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def layernorm_residual(x, residual, weight, bias):
    x = x + residual
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var  = jnp.var(x,  axis=-1, keepdims=True)
    return weight * (x - mean) / jnp.sqrt(var + 1e-5) + bias
""",
)

_p34 = Problem(
    task_id=34, name="swiglu", level=2, category="fused",
    **_urls("kernels/level2/swiglu.py"),
    input_shapes=[(512, 2048)], input_dtypes=["float32"],
    jax_functional="""
def swiglu(x):
    import jax, jax.numpy as jnp
    d = x.shape[-1] // 2
    gate, val = x[..., :d], x[..., d:]
    return val * jax.nn.silu(gate)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax
        d = x.shape[-1] // 2
        gate, val = x[..., :d], x[..., d:]
        return val * jax.nn.silu(gate)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def swiglu_kernel(x_ref, o_ref):
    x = x_ref[...]                       # shape (bm, D)
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    o_ref[...] = val * (gate * jax.nn.sigmoid(gate))

def swiglu(x):
    B, D = x.shape; bm = min(16, B)
    half = D // 2
    return pl.pallas_call(
        swiglu_kernel,
        out_shape=jax.ShapeDtypeStruct((B, half), x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, D), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm, half), lambda i: (i, 0)),
    )(x)
""",
)

_p35 = Problem(
    task_id=35, name="geglu", level=2, category="fused",
    **_urls("kernels/level2/geglu.py"),
    input_shapes=[(512, 2048)], input_dtypes=["float32"],
    jax_functional="""
def geglu(x):
    import jax
    d = x.shape[-1] // 2
    gate, val = x[..., :d], x[..., d:]
    return val * jax.nn.gelu(gate)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax
        d = x.shape[-1] // 2
        gate, val = x[..., :d], x[..., d:]
        return val * jax.nn.gelu(gate)
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def geglu_kernel(x_ref, o_ref):
    x = x_ref[...]                       # shape (bm, D)
    half = x.shape[-1] // 2
    gate, val = x[..., :half], x[..., half:]
    c = 0.7978845608028654
    gelu_gate = 0.5 * gate * (1.0 + jnp.tanh(c * (gate + 0.044715 * gate**3)))
    o_ref[...] = val * gelu_gate

def geglu(x):
    B, D = x.shape; bm = min(16, B)
    half = D // 2
    return pl.pallas_call(
        geglu_kernel,
        out_shape=jax.ShapeDtypeStruct((B, half), x.dtype),
        grid=(B // bm,),
        in_specs=[pl.BlockSpec((bm, D), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((bm, half), lambda i: (i, 0)),
    )(x)
""",
)

_p36 = Problem(
    task_id=36, name="qk_softmax", level=2, category="attention",
    **_urls("kernels/level2/qk_softmax.py"),
    input_shapes=[(8, 16, 64), (8, 16, 64)], input_dtypes=["float32"]*2,
    jax_functional="""
def qk_softmax(q, k):
    import jax, jax.numpy as jnp
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum('...hd,...kd->...hk', q, k) * scale
    return jax.nn.softmax(scores, axis=-1)
""",
    jax_module="""
class Model:
    def __call__(self, q, k):
        import jax, jax.numpy as jnp
        scale = q.shape[-1] ** -0.5
        scores = jnp.einsum('bhd,bkd->bhk', q, k) * scale
        return jax.nn.softmax(scores, axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

def qk_softmax(q, k):
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum('bhd,bkd->bhk', q, k) * scale
    return jax.nn.softmax(scores, axis=-1)
""",
)

_p37 = Problem(
    task_id=37, name="fused_softmax_cross_entropy", level=2, category="loss",
    **_urls("kernels/level2/fused_softmax_cross_entropy.py"),
    input_shapes=[(512, 1024), (512,)], input_dtypes=["float32", "int32"],
    jax_functional="""
def fused_softmax_cross_entropy(logits, labels):
    import jax, jax.numpy as jnp
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
""",
    jax_module="""
class Model:
    def __call__(self, logits, labels):
        import jax, jax.numpy as jnp
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
""",
    seed_pallas="""
import jax, jax.numpy as jnp

def fused_softmax_cross_entropy(logits, labels):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
""",
)

_p38 = Problem(
    task_id=38, name="sigmoid_bce", level=2, category="loss",
    **_urls("kernels/level2/sigmoid_bce.py"),
    input_shapes=[(4096,), (4096,)], input_dtypes=["float32"]*2,
    jax_functional="""
def sigmoid_bce(logits, targets):
    import jax.numpy as jnp
    p = 1.0 / (1.0 + jnp.exp(-logits))
    return -jnp.mean(targets * jnp.log(p + 1e-7) + (1.0 - targets) * jnp.log(1.0 - p + 1e-7))
""",
    jax_module="""
class Model:
    def __call__(self, logits, targets):
        import jax.numpy as jnp
        p = jax.nn.sigmoid(logits)
        return -jnp.mean(targets * jnp.log(p + 1e-7) + (1-targets) * jnp.log(1-p + 1e-7))
""",
    seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl

def bce_kernel(logits_ref, tgt_ref, o_ref):
    logits = logits_ref[...]
    tgt    = tgt_ref[...]
    p = 1.0 / (1.0 + jnp.exp(-logits))
    o_ref[...] = -(tgt * jnp.log(p + 1e-7) + (1.0 - tgt) * jnp.log(1.0 - p + 1e-7))

def sigmoid_bce(logits, targets):
    n = logits.shape[0]; block = min(2048, n)
    losses = pl.pallas_call(
        bce_kernel,
        out_shape=jax.ShapeDtypeStruct(logits.shape, logits.dtype),
        grid=(n // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,)),
                  pl.BlockSpec((block,), lambda i: (i,))],
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(logits, targets)
    return jnp.mean(losses)
""",
)

_p39 = Problem(
    task_id=39, name="pwm_scan", level=2, category="genomics",
    **_urls("kernels/level2/pwm_scan.py"),
    input_shapes=[(256, 4, 20), (4, 20)], input_dtypes=["float32"]*2,
    jax_functional="""
def pwm_scan(seq_onehot, pwm):
    import jax.numpy as jnp
    # seq_onehot: (B, 4, L), pwm: (4, W) -> scores: (B, L-W+1)
    B, C, L = seq_onehot.shape; W = pwm.shape[1]
    scores = jnp.stack([
        jnp.sum(seq_onehot[:, :, i:i+W] * pwm, axis=(1,2))
        for i in range(L - W + 1)
    ], axis=-1)
    return scores
""",
    jax_module="""
class Model:
    def __call__(self, seq, pwm):
        import jax.numpy as jnp
        B, C, L = seq.shape; W = pwm.shape[1]
        return jnp.stack([jnp.sum(seq[:, :, i:i+W] * pwm, axis=(1,2))
                          for i in range(L - W + 1)], axis=-1)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: pwm_scan requires scan with non-array state — falls back to JAX.
def pwm_scan(seq_onehot, pwm):
    B, C, L = seq_onehot.shape; W = pwm.shape[1]
    return jnp.stack([jnp.sum(seq_onehot[:, :, i:i+W] * pwm, axis=(1,2))
                      for i in range(L - W + 1)], axis=-1)
""",
)

_p40 = Problem(
    task_id=40, name="pairwise_distance", level=2, category="genomics",
    **_urls("kernels/level2/pairwise_distance.py"),
    input_shapes=[(256, 128)], input_dtypes=["float32"],
    jax_functional="""
def pairwise_distance(x):
    import jax.numpy as jnp
    diff = x[:, None, :] - x[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
""",
    jax_module="""
class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        diff = x[:, None, :] - x[None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: pairwise_distance requires nested reductions — falls back to JAX.
def pairwise_distance(x):
    diff = x[:, None, :] - x[None, :, :]
    return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
""",
)

# ===========================================================================
# LEVEL 3 — Architecture Components (5 problems, task_ids 41–45)
# ===========================================================================

_p41 = Problem(
    task_id=41, name="flash_attention", level=3, category="attention",
    **_urls("kernels/level3/flash_attention.py"),
    input_shapes=[(2, 8, 512, 64), (2, 8, 512, 64), (2, 8, 512, 64)],
    input_dtypes=["float16"]*3,
    jax_functional="""
def flash_attention(q, k, v):
    import jax, jax.numpy as jnp
    scale = q.shape[-1] ** -0.5
    scores = jnp.einsum('...hqd,...hkd->...hqk', q, k) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('...hqk,...hkd->...hqd', weights, v)
""",
    jax_module="""
class Model:
    def __call__(self, q, k, v):
        import jax, jax.numpy as jnp
        scale = q.shape[-1] ** -0.5
        scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        return jnp.einsum('bhqk,bhkd->bhqd', jax.nn.softmax(scores, axis=-1), v)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

def flash_attention(q, k, v):
    # Cast to float32 for numerics, return in original dtype
    orig_dtype = q.dtype
    q32, k32, v32 = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    scale = q32.shape[-1] ** -0.5
    scores = jnp.einsum('...hqd,...hkd->...hqk', q32, k32) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('...hqk,...hkd->...hqd', weights, v32)
    return out.astype(orig_dtype)
""",
)

_p42 = Problem(
    task_id=42, name="multi_head_attention", level=3, category="attention",
    **_urls("kernels/level3/multi_head_attention.py"),
    input_shapes=[(2, 512, 512), (512, 512), (512, 512), (512, 512), (512, 512)],
    input_dtypes=["float32"] * 5,
    jax_functional="""
def multi_head_attention(x, wq, wk, wv, wo, n_heads=8):
    import jax, jax.numpy as jnp
    B, T, D = x.shape; H = n_heads; d = D // H
    q = (x @ wq).reshape(B, T, H, d).transpose(0,2,1,3)
    k = (x @ wk).reshape(B, T, H, d).transpose(0,2,1,3)
    v = (x @ wv).reshape(B, T, H, d).transpose(0,2,1,3)
    scale = d ** -0.5
    att = jax.nn.softmax(jnp.einsum('...hqd,...hkd->...hqk', q, k) * scale, axis=-1)
    out = jnp.einsum('...hqk,...hkd->...hqd', att, v).transpose(0,2,1,3).reshape(B, T, D)
    return out @ wo
""",
    jax_module="""
class Model:
    def __call__(self, x, wq, wk, wv, wo):
        import jax, jax.numpy as jnp
        B, T, D = x.shape; H, d = 8, D // 8
        q = (x @ wq).reshape(B, T, H, d).transpose(0,2,1,3)
        k = (x @ wk).reshape(B, T, H, d).transpose(0,2,1,3)
        v = (x @ wv).reshape(B, T, H, d).transpose(0,2,1,3)
        att = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, k) * d**-0.5, axis=-1)
        return jnp.einsum('bhqk,bhkd->bhqd', att, v).transpose(0,2,1,3).reshape(B,T,D) @ wo
""",
    seed_pallas="""
import jax, jax.numpy as jnp

def multi_head_attention(x, wq, wk, wv, wo, n_heads=8):
    B, T, D = x.shape; H, d = n_heads, D // n_heads
    q = (x @ wq).reshape(B, T, H, d).transpose(0,2,1,3)
    k = (x @ wk).reshape(B, T, H, d).transpose(0,2,1,3)
    v = (x @ wv).reshape(B, T, H, d).transpose(0,2,1,3)
    scale = d ** -0.5
    att = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', att, v).transpose(0,2,1,3).reshape(B, T, D)
    return out @ wo
""",
)

_p43 = Problem(
    task_id=43, name="gated_mlp", level=3, category="mlp",
    **_urls("kernels/level3/gated_mlp.py"),
    input_shapes=[(4, 512, 512), (512, 2048), (2048, 512)], input_dtypes=["float32"]*3,
    jax_functional="""
def gated_mlp(x, w1, w2):
    import jax
    hidden = jax.nn.silu(x @ w1)
    return hidden @ w2
""",
    jax_module="""
class Model:
    def __call__(self, x, w1, w2):
        import jax
        return jax.nn.silu(x @ w1) @ w2
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: gated_mlp with weight matrices >1M elements — falls back to JAX.
def gated_mlp(x, w1, w2):
    return jax.nn.silu(x @ w1) @ w2
""",
)

_p44 = Problem(
    task_id=44, name="transformer_block", level=3, category="transformer",
    **_urls("kernels/level3/transformer_block.py"),
    input_shapes=[(2, 128, 512), (512,512), (512,512), (512,512), (512,512),
                  (512, 2048), (2048, 512), (512,), (512,)],
    input_dtypes=["float32"] * 9,
    jax_functional="""
def transformer_block(x, wq, wk, wv, wo, w1, w2, norm_w1, norm_w2):
    import jax, jax.numpy as jnp
    def rmsnorm(h, w):
        rms = jnp.sqrt(jnp.mean(h**2, axis=-1, keepdims=True) + 1e-6)
        return w * h / rms
    # Attention sub-layer
    h = rmsnorm(x, norm_w1)
    B, T, D = h.shape; H, d = 8, D // 8
    q = (h @ wq).reshape(B,T,H,d).transpose(0,2,1,3)
    k = (h @ wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v = (h @ wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk', q, k) * d**-0.5, axis=-1)
    attn_out = jnp.einsum('bhqk,bhkd->bhqd', att, v).transpose(0,2,1,3).reshape(B,T,D) @ wo
    x = x + attn_out
    # FFN sub-layer
    h = rmsnorm(x, norm_w2)
    x = x + jax.nn.silu(h @ w1) @ w2
    return x
""",
    jax_module="""
class Model:
    def __call__(self, x, *weights):
        import jax, jax.numpy as jnp
        wq, wk, wv, wo, w1, w2, nw1, nw2 = weights
        def rmsnorm(h, w):
            return w * h / jnp.sqrt(jnp.mean(h**2, axis=-1, keepdims=True) + 1e-6)
        B, T, D = x.shape; H, d = 8, D//8
        h = rmsnorm(x, nw1)
        q = (h@wq).reshape(B,T,H,d).transpose(0,2,1,3)
        k = (h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
        v = (h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5, axis=-1)
        x = x + jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        h = rmsnorm(x, nw2)
        return x + jax.nn.silu(h@w1)@w2
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: full transformer_block — falls back to JAX (complex decomposition needed).
def transformer_block(x, wq, wk, wv, wo, w1, w2, norm_w1, norm_w2):
    def rmsnorm(h, w):
        return w * h / jnp.sqrt(jnp.mean(h**2, axis=-1, keepdims=True) + 1e-6)
    B, T, D = x.shape; H, d = 8, D // 8
    h = rmsnorm(x, norm_w1)
    q = (h@wq).reshape(B,T,H,d).transpose(0,2,1,3)
    k = (h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v = (h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att = jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5, axis=-1)
    x = x + jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    h = rmsnorm(x, norm_w2)
    return x + jax.nn.silu(h@w1)@w2
""",
)

_p45 = Problem(
    task_id=45, name="triangle_update", level=3, category="transformer",
    **_urls("kernels/level3/triangle_update.py"),
    input_shapes=[(64, 64, 64), (64, 64, 64), (64, 64, 1)], input_dtypes=["float32"]*3,
    jax_functional="""
def triangle_update(z, mask, bias):
    import jax, jax.numpy as jnp
    # Simplified AlphaFold2 triangular update (outgoing)
    a = jax.nn.sigmoid(z + bias) * mask
    b = jax.nn.sigmoid(z + bias) * mask
    return jnp.einsum('...ikc,...jkc->...ijc', a, b)
""",
    jax_module="""
class Model:
    def __call__(self, z, mask, bias):
        import jax, jax.numpy as jnp
        a = jax.nn.sigmoid(z + bias) * mask
        b = jax.nn.sigmoid(z + bias) * mask
        return jnp.einsum('ikc,jkc->ijc', a, b)
""",
    seed_pallas="""
import jax, jax.numpy as jnp

# NOTE: triangle_update requires nested reduction — falls back to JAX.
def triangle_update(z, mask, bias):
    a = jax.nn.sigmoid(z + bias) * mask
    b = jax.nn.sigmoid(z + bias) * mask
    return jnp.einsum('ikc,jkc->ijc', a, b)
""",
)


# ===========================================================================
# Registry
# ===========================================================================

ALL_PROBLEMS: list[Problem] = [
    _p1, _p2, _p3, _p4, _p5,           # L1: activations
    _p6, _p7,                            # L1: normalization
    _p8, _p9, _p10,                      # L1: matmul
    _p11, _p12, _p13,                    # L1: reductions
    _p14, _p15,                          # L1: softmax
    _p16, _p17, _p18,                    # L1: transcendentals
    _p19, _p20, _p21,                    # L1: elementwise
    _p22, _p23, _p24,                    # L1: loss
    _p25, _p26, _p27,                    # L1: data/index
    _p28, _p29, _p30, _p31,              # L2: matmul fusions
    _p32, _p33, _p34, _p35,              # L2: norm + gating
    _p36, _p37, _p38, _p39, _p40,       # L2: attention + loss + genomics
    _p41, _p42, _p43, _p44, _p45,       # L3: architecture
]

BY_ID: dict[int, Problem] = {p.task_id: p for p in ALL_PROBLEMS}
BY_NAME: dict[str, Problem] = {p.name: p for p in ALL_PROBLEMS}

LEVEL_PROBLEMS: dict[int, list[Problem]] = {
    1: [p for p in ALL_PROBLEMS if p.level == 1],
    2: [p for p in ALL_PROBLEMS if p.level == 2],
    3: [p for p in ALL_PROBLEMS if p.level == 3],
}

assert len(ALL_PROBLEMS) == 45, f"Expected 45 problems, got {len(ALL_PROBLEMS)}"
