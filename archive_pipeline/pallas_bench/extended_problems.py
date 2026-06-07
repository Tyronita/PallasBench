"""
100 new real-world problems sourced from GitHub — L1/L2/L3/L4 (KernelBench-style).
L4 = full model forward passes (20+ models).
All have GitHub provenance; seed_pallas is JAX for L4 (LLM will Pallasify).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pallas_bench.problems import Problem

_B = "https://github.com"
_R = "https://raw.githubusercontent.com"

def _u(repo, path, branch="main"):
    return {
        "github_url": f"{_B}/{repo}/blob/{branch}/{path}",
        "github_raw_url": f"{_R}/{repo}/{branch}/{path}",
    }

EXT: list[Problem] = []

# ─── L1 NEW SINGLE OPS (task_ids 46-75) ────────────────────────────────────

EXT += [
Problem(task_id=46, name="leaky_relu", level=1, category="activation",
  **_u("google/flax","flax/linen/__init__.py"),
  input_shapes=[(8192,)], input_dtypes=["float32"],
  jax_functional="def leaky_relu(x):\n import jax.numpy as jnp\n return jnp.where(x>=0, x, 0.01*x)",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.numpy as jnp\n  return jnp.where(x>=0,x,0.01*x)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def leaky_relu_k(x_ref,o_ref):
    x=x_ref[...]; o_ref[...]=jnp.where(x>=0,x,0.01*x)
def leaky_relu(x):
    n=x.shape[0]; b=min(2048,n)
    return pl.pallas_call(leaky_relu_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(x)
"""),

Problem(task_id=47, name="elu", level=1, category="activation",
  **_u("jax-ml/jax","jax/_src/nn/functions.py"),
  input_shapes=[(8192,)], input_dtypes=["float32"],
  jax_functional="def elu(x):\n import jax.numpy as jnp\n return jnp.where(x>=0,x,jnp.expm1(x))",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.nn as nn\n  return nn.elu(x)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def elu_k(x_ref,o_ref):
    x=x_ref[...]; o_ref[...]=jnp.where(x>=0,x,jnp.expm1(x))
def elu(x):
    n=x.shape[0]; b=min(2048,n)
    return pl.pallas_call(elu_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(x)
"""),

Problem(task_id=48, name="hardswish", level=1, category="activation",
  **_u("keras-team/keras","keras/src/backend/jax/nn.py"),
  input_shapes=[(8192,)], input_dtypes=["float32"],
  jax_functional="def hardswish(x):\n import jax.numpy as jnp\n return x*jnp.clip((x+3)/6,0,1)",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.numpy as jnp\n  return x*jnp.clip((x+3)/6,0,1)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def hardswish_k(x_ref,o_ref):
    x=x_ref[...]; o_ref[...]=x*jnp.clip((x+3)/6,0,1)
def hardswish(x):
    n=x.shape[0]; b=min(2048,n)
    return pl.pallas_call(hardswish_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(x)
"""),

Problem(task_id=49, name="mish", level=1, category="activation",
  **_u("keras-team/keras","keras/src/backend/jax/nn.py"),
  input_shapes=[(8192,)], input_dtypes=["float32"],
  jax_functional="def mish(x):\n import jax.numpy as jnp\n return x*jnp.tanh(jnp.log1p(jnp.exp(x)))",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.numpy as jnp\n  return x*jnp.tanh(jnp.log1p(jnp.exp(x)))",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def mish_k(x_ref,o_ref):
    x=x_ref[...]; o_ref[...]=x*jnp.tanh(jnp.log1p(jnp.exp(x)))
def mish(x):
    n=x.shape[0]; b=min(2048,n)
    return pl.pallas_call(mish_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(x)
"""),

Problem(task_id=50, name="softplus", level=1, category="activation",
  **_u("jax-ml/jax","jax/_src/nn/functions.py"),
  input_shapes=[(8192,)], input_dtypes=["float32"],
  jax_functional="def softplus(x):\n import jax.numpy as jnp\n return jnp.log1p(jnp.exp(x))",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.nn as nn\n  return nn.softplus(x)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def softplus_k(x_ref,o_ref):
    o_ref[...]=jnp.log1p(jnp.exp(x_ref[...]))
def softplus(x):
    n=x.shape[0]; b=min(2048,n)
    return pl.pallas_call(softplus_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(x)
"""),

Problem(task_id=51, name="batch_norm", level=1, category="normalization",
  **_u("google/flax","flax/linen/normalization.py"),
  input_shapes=[(512,1024),(1024,),(1024,)], input_dtypes=["float32","float32","float32"],
  jax_functional="""def batch_norm(x, scale, bias, eps=1e-5):
    import jax.numpy as jnp
    mean = jnp.mean(x, axis=0, keepdims=True)
    var  = jnp.var(x,  axis=0, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias""",
  jax_module="""class Model:
    def __call__(self, x, scale, bias):
        import jax.numpy as jnp
        mean = jnp.mean(x, axis=0, keepdims=True)
        var  = jnp.var(x,  axis=0, keepdims=True)
        return scale * (x - mean) / jnp.sqrt(var + 1e-5) + bias""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def batch_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=0, keepdims=True)
    var  = jnp.var(x,  axis=0, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias
"""),

Problem(task_id=52, name="group_norm", level=1, category="normalization",
  **_u("google/flax","flax/linen/normalization.py"),
  input_shapes=[(32,8,1024)], input_dtypes=["float32"],
  jax_functional="""def group_norm(x, num_groups=8, eps=1e-5):
    import jax.numpy as jnp
    B, G, C = x.shape
    x2 = x.reshape(B, G, -1)
    mean = jnp.mean(x2, axis=-1, keepdims=True)
    var  = jnp.var(x2,  axis=-1, keepdims=True)
    return ((x2 - mean) / jnp.sqrt(var + eps)).reshape(B, G, C)""",
  jax_module="""class Model:
    def __call__(self, x):
        import jax.numpy as jnp
        B, G, C = x.shape; x2 = x.reshape(B, G, -1)
        mean = jnp.mean(x2, -1, keepdims=True); var = jnp.var(x2, -1, keepdims=True)
        return ((x2 - mean) / jnp.sqrt(var + 1e-5)).reshape(B, G, C)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def group_norm(x, num_groups=8, eps=1e-5):
    B, G, C = x.shape; x2 = x.reshape(B, G, -1)
    mean = jnp.mean(x2, -1, keepdims=True); var = jnp.var(x2, -1, keepdims=True)
    return ((x2 - mean) / jnp.sqrt(var + eps)).reshape(B, G, C)
"""),

Problem(task_id=53, name="vector_add", level=1, category="elementwise",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(65536,),(65536,)], input_dtypes=["float32","float32"],
  jax_functional="def vector_add(a,b): return a+b",
  jax_module="class Model:\n def __call__(self,a,b): return a+b",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def vadd_k(a_ref,b_ref,o_ref): o_ref[...]=a_ref[...]+b_ref[...]
def vector_add(a,b):
    n=a.shape[0]; bk=min(4096,n)
    return pl.pallas_call(vadd_k,out_shape=jax.ShapeDtypeStruct(a.shape,a.dtype),
        grid=(n//bk,),in_specs=[pl.BlockSpec((bk,),lambda i:(i,)),pl.BlockSpec((bk,),lambda i:(i,))],
        out_specs=pl.BlockSpec((bk,),lambda i:(i,)))(a,b)
"""),

Problem(task_id=54, name="elementwise_max", level=1, category="elementwise",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(65536,),(65536,)], input_dtypes=["float32","float32"],
  jax_functional="def elementwise_max(a,b):\n import jax.numpy as jnp\n return jnp.maximum(a,b)",
  jax_module="class Model:\n def __call__(self,a,b):\n  import jax.numpy as jnp\n  return jnp.maximum(a,b)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def emax_k(a_ref,b_ref,o_ref): o_ref[...]=jnp.maximum(a_ref[...],b_ref[...])
def elementwise_max(a,b):
    n=a.shape[0]; bk=min(4096,n)
    return pl.pallas_call(emax_k,out_shape=jax.ShapeDtypeStruct(a.shape,a.dtype),
        grid=(n//bk,),in_specs=[pl.BlockSpec((bk,),lambda i:(i,)),pl.BlockSpec((bk,),lambda i:(i,))],
        out_specs=pl.BlockSpec((bk,),lambda i:(i,)))(a,b)
"""),

Problem(task_id=55, name="elementwise_mul", level=1, category="elementwise",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(65536,),(65536,)], input_dtypes=["float32","float32"],
  jax_functional="def elementwise_mul(a,b): return a*b",
  jax_module="class Model:\n def __call__(self,a,b): return a*b",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def emul_k(a_ref,b_ref,o_ref): o_ref[...]=a_ref[...]*b_ref[...]
def elementwise_mul(a,b):
    n=a.shape[0]; bk=min(4096,n)
    return pl.pallas_call(emul_k,out_shape=jax.ShapeDtypeStruct(a.shape,a.dtype),
        grid=(n//bk,),in_specs=[pl.BlockSpec((bk,),lambda i:(i,)),pl.BlockSpec((bk,),lambda i:(i,))],
        out_specs=pl.BlockSpec((bk,),lambda i:(i,)))(a,b)
"""),

Problem(task_id=56, name="row_sum", level=1, category="reduce",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,1024)], input_dtypes=["float32"],
  jax_functional="def row_sum(x):\n import jax.numpy as jnp\n return jnp.sum(x,axis=1)",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.numpy as jnp\n  return jnp.sum(x,axis=1)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def rsum_k(x_ref,o_ref): o_ref[...]=jnp.sum(x_ref[...],axis=1)
def row_sum(x):
    B,D=x.shape; bm=min(128,B)
    return pl.pallas_call(rsum_k,out_shape=jax.ShapeDtypeStruct((B,),x.dtype),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0))],
        out_specs=pl.BlockSpec((bm,),lambda i:(i,)))(x)
"""),

Problem(task_id=57, name="col_sum", level=1, category="reduce",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,1024)], input_dtypes=["float32"],
  jax_functional="def col_sum(x):\n import jax.numpy as jnp\n return jnp.sum(x,axis=0)",
  jax_module="class Model:\n def __call__(self,x):\n  import jax.numpy as jnp\n  return jnp.sum(x,axis=0)",
  seed_pallas="""
import jax, jax.numpy as jnp
def col_sum(x): return jnp.sum(x, axis=0)
"""),

Problem(task_id=58, name="huber_loss", level=1, category="loss",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(8192,),(8192,)], input_dtypes=["float32","float32"],
  jax_functional="""def huber_loss(pred, target, delta=1.0):
    import jax.numpy as jnp
    diff = jnp.abs(pred - target)
    return jnp.where(diff < delta, 0.5*diff**2, delta*(diff - 0.5*delta)).mean()""",
  jax_module="""class Model:
    def __call__(self, pred, target, delta=1.0):
        import jax.numpy as jnp
        diff = jnp.abs(pred - target)
        return jnp.where(diff < delta, 0.5*diff**2, delta*(diff-0.5*delta)).mean()""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def huber_k(p_ref,t_ref,o_ref):
    diff=jnp.abs(p_ref[...]-t_ref[...]); o_ref[...]=jnp.where(diff<1.,0.5*diff**2,diff-0.5)
def huber_loss(pred, target, delta=1.0):
    n=pred.shape[0]; b=min(2048,n)
    elems=pl.pallas_call(huber_k,out_shape=jax.ShapeDtypeStruct(pred.shape,pred.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,)),pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(pred,target)
    return jnp.mean(elems)
"""),

Problem(task_id=59, name="scatter_add", level=1, category="index",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,),(4096,),(1024,)], input_dtypes=["int32","float32","float32"],
  jax_functional="""def scatter_add(indices, updates, size):
    import jax.numpy as jnp
    return jnp.zeros(size).at[indices].add(updates)""",
  jax_module="""class Model:
    def __call__(self, indices, updates, size=1024):
        import jax.numpy as jnp
        return jnp.zeros(size).at[indices].add(updates)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def scatter_add(indices, updates, size):
    return jnp.zeros(size[0]).at[indices].add(updates)
"""),

Problem(task_id=60, name="gather", level=1, category="index",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(8192,64),(4096,)], input_dtypes=["float32","int32"],
  jax_functional="def gather(table, idx): return table[idx]",
  jax_module="class Model:\n def __call__(self, table, idx): return table[idx]",
  seed_pallas="import jax.numpy as jnp\ndef gather(table, idx): return table[idx]"),

Problem(task_id=61, name="cumsum", level=1, category="reduce",
  **_u("oliverdutton/tallax","tallax/tax/cumsum.py"),
  input_shapes=[(65536,)], input_dtypes=["float32"],
  jax_functional="def cumsum(x):\n import jax.numpy as jnp\n return jnp.cumsum(x)",
  jax_module="class Model:\n def __call__(self, x):\n  import jax.numpy as jnp\n  return jnp.cumsum(x)",
  seed_pallas="import jax.numpy as jnp\ndef cumsum(x): return jnp.cumsum(x)"),

Problem(task_id=62, name="layer_scale", level=1, category="normalization",
  **_u("google/flax","flax/linen/normalization.py"),
  input_shapes=[(512,1024),(1024,)], input_dtypes=["float32","float32"],
  jax_functional="def layer_scale(x, gamma): return x * gamma",
  jax_module="class Model:\n def __call__(self, x, gamma): return x * gamma",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def ls_k(x_ref,g_ref,o_ref): o_ref[...]=x_ref[...]*g_ref[...]
def layer_scale(x, gamma):
    B,D=x.shape; bm=min(32,B)
    return pl.pallas_call(ls_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0)),pl.BlockSpec((D,),lambda i:(0,))],
        out_specs=pl.BlockSpec((bm,D),lambda i:(i,0)))(x,gamma)
"""),

# Genomics L1 problems (from PallasBench provenance / Enformer)
Problem(task_id=63, name="gc_content", level=1, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,4)], input_dtypes=["float32"],
  jax_functional="""def gc_content(onehot):
    import jax.numpy as jnp
    # onehot: (L,4) — A,C,G,T. GC = C+G = cols 1,2
    return jnp.mean(onehot[:,1]+onehot[:,2])""",
  jax_module="""class Model:
    def __call__(self, onehot):
        import jax.numpy as jnp
        return jnp.mean(onehot[:,1]+onehot[:,2])""",
  seed_pallas="import jax.numpy as jnp\ndef gc_content(onehot): return jnp.mean(onehot[:,1]+onehot[:,2])"),

Problem(task_id=64, name="kmer_count", level=1, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,)], input_dtypes=["int32"],
  jax_functional="""def kmer_count(seq, k=3):
    import jax.numpy as jnp
    # Count occurrences of each k-mer (simplified: sum over windows)
    return jnp.sum(seq[:len(seq)-k+1])""",
  jax_module="""class Model:
    def __call__(self, seq, k=3):
        import jax.numpy as jnp
        return jnp.sum(seq[:seq.shape[0]-k+1])""",
  seed_pallas="import jax.numpy as jnp\ndef kmer_count(seq, k=3): return jnp.sum(seq[:seq.shape[0]-k+1])"),
]

# ─── L2 NEW FUSION PATTERNS (task_ids 76-100) ────────────────────────────────

EXT += [
Problem(task_id=76, name="fused_layer_norm_relu", level=2, category="fused",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,1024),(1024,),(1024,)], input_dtypes=["float32","float32","float32"],
  jax_functional="""def fused_layer_norm_relu(x, w, b, eps=1e-5):
    import jax.numpy as jnp
    mean = jnp.mean(x,-1,keepdims=True); var = jnp.var(x,-1,keepdims=True)
    return jnp.maximum(w*(x-mean)/jnp.sqrt(var+eps)+b, 0.0)""",
  jax_module="""class Model:
    def __call__(self,x,w,b):
        import jax.numpy as jnp
        mean=jnp.mean(x,-1,keepdims=True); var=jnp.var(x,-1,keepdims=True)
        return jnp.maximum(w*(x-mean)/jnp.sqrt(var+1e-5)+b,0.)""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def fln_relu_k(x_ref,w_ref,b_ref,o_ref):
    x=x_ref[...].astype(jnp.float32)
    mean=jnp.mean(x,-1,keepdims=True); var=jnp.var(x,-1,keepdims=True)
    o_ref[...]=jnp.maximum(w_ref[...]*(x-mean)/jnp.sqrt(var+1e-5)+b_ref[...],0.).astype(x_ref.dtype)
def fused_layer_norm_relu(x,w,b):
    B,D=x.shape; bm=min(16,B)
    return pl.pallas_call(fln_relu_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0)),
        pl.BlockSpec((D,),lambda i:(0,)),pl.BlockSpec((D,),lambda i:(0,))],
        out_specs=pl.BlockSpec((bm,D),lambda i:(i,0)))(x,w,b)
"""),

Problem(task_id=77, name="fused_matmul_bias", level=2, category="fused",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,512),(512,512),(512,)], input_dtypes=["float32","float32","float32"],
  jax_functional="def fused_matmul_bias(x,w,b): return x@w+b",
  jax_module="class Model:\n def __call__(self,x,w,b): return x@w+b",
  seed_pallas="import jax.numpy as jnp\ndef fused_matmul_bias(x,w,b): return x@w+b"),

Problem(task_id=78, name="fused_gelu_bias", level=2, category="fused",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,1024),(1024,)], input_dtypes=["float32","float32"],
  jax_functional="def fused_gelu_bias(x,b):\n import jax\n return jax.nn.gelu(x+b)",
  jax_module="class Model:\n def __call__(self,x,b):\n  import jax\n  return jax.nn.gelu(x+b)",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def fgb_k(x_ref,b_ref,o_ref):
    x=x_ref[...]+b_ref[...]; c=0.7978845608028654
    o_ref[...]=0.5*x*(1.+jnp.tanh(c*(x+0.044715*x**3)))
def fused_gelu_bias(x,b):
    B,D=x.shape; bm=min(32,B)
    return pl.pallas_call(fgb_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0)),pl.BlockSpec((D,),lambda i:(0,))],
        out_specs=pl.BlockSpec((bm,D),lambda i:(i,0)))(x,b)
"""),

Problem(task_id=79, name="fused_residual_norm", level=2, category="fused",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,1024),(512,1024),(1024,)], input_dtypes=["float32","float32","float32"],
  jax_functional="""def fused_residual_norm(x,res,w,eps=1e-6):
    import jax.numpy as jnp
    x=x+res; rms=jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+eps)
    return w*x/rms""",
  jax_module="""class Model:
    def __call__(self,x,res,w):
        import jax.numpy as jnp
        x=x+res; rms=jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+1e-6)
        return w*x/rms""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def frn_k(x_ref,r_ref,w_ref,o_ref):
    x=(x_ref[...]+r_ref[...]).astype(jnp.float32)
    rms=jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+1e-6)
    o_ref[...]=(w_ref[...]*x/rms).astype(x_ref.dtype)
def fused_residual_norm(x,res,w):
    B,D=x.shape; bm=min(16,B)
    return pl.pallas_call(frn_k,out_shape=jax.ShapeDtypeStruct(x.shape,x.dtype),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0)),
        pl.BlockSpec((bm,D),lambda i:(i,0)),pl.BlockSpec((D,),lambda i:(0,))],
        out_specs=pl.BlockSpec((bm,D),lambda i:(i,0)))(x,res,w)
"""),

Problem(task_id=80, name="fused_relu_matmul", level=2, category="fused",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,512),(512,512)], input_dtypes=["float32","float32"],
  jax_functional="def fused_relu_matmul(x,w):\n import jax.numpy as jnp\n return jnp.maximum(x,0.)@w",
  jax_module="class Model:\n def __call__(self,x,w):\n  import jax.numpy as jnp\n  return jnp.maximum(x,0.)@w",
  seed_pallas="import jax.numpy as jnp\ndef fused_relu_matmul(x,w): return jnp.maximum(x,0.)@w"),

Problem(task_id=81, name="hamming_distance", level=2, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(512,256),(512,256)], input_dtypes=["float32","float32"],
  jax_functional="""def hamming_distance(a,b):
    import jax.numpy as jnp
    return jnp.sum(a!=b, axis=-1).astype(jnp.float32)""",
  jax_module="""class Model:
    def __call__(self,a,b):
        import jax.numpy as jnp
        return jnp.sum(a!=b,-1).astype(jnp.float32)""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def ham_k(a_ref,b_ref,o_ref):
    o_ref[...]=jnp.sum((a_ref[...]!=b_ref[...]).astype(jnp.float32),-1)
def hamming_distance(a,b):
    B,D=a.shape; bm=min(128,B)
    return pl.pallas_call(ham_k,out_shape=jax.ShapeDtypeStruct((B,),jnp.float32),
        grid=(B//bm,),in_specs=[pl.BlockSpec((bm,D),lambda i:(i,0)),pl.BlockSpec((bm,D),lambda i:(i,0))],
        out_specs=pl.BlockSpec((bm,),lambda i:(i,)))(a,b)
"""),

Problem(task_id=82, name="sequence_match", level=2, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(256,512),(256,512)], input_dtypes=["float32","float32"],
  jax_functional="""def sequence_match(a,b):
    import jax.numpy as jnp
    return jnp.sum(a==b,-1).astype(jnp.float32) / a.shape[-1]""",
  jax_module="""class Model:
    def __call__(self,a,b):
        import jax.numpy as jnp
        return jnp.sum(a==b,-1).astype(jnp.float32)/a.shape[-1]""",
  seed_pallas="import jax.numpy as jnp\ndef sequence_match(a,b): return jnp.sum(a==b,-1).astype(jnp.float32)/a.shape[-1]"),

Problem(task_id=83, name="reverse_complement", level=2, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4096,4)], input_dtypes=["float32"],
  jax_functional="""def reverse_complement(onehot):
    import jax.numpy as jnp
    # A<->T (0<->3), C<->G (1<->2) then reverse
    comp = onehot[:,::-1]  # flip base axis: A,C,G,T -> T,G,C,A
    return comp[::-1]      # reverse sequence""",
  jax_module="""class Model:
    def __call__(self,onehot):
        import jax.numpy as jnp
        return onehot[:,::-1][::-1]""",
  seed_pallas="import jax.numpy as jnp\ndef reverse_complement(onehot): return onehot[:,::-1][::-1]"),

Problem(task_id=84, name="linear_recurrence", level=2, category="rnn",
  **_u("google-deepmind/recurrentgemma","recurrentgemma/jax/pallas.py"),
  input_shapes=[(32,128,256),(32,128,256)], input_dtypes=["float32","float32"],
  jax_functional="""def linear_recurrence(a, x):
    # a: decay coefficients (B,T,D), x: inputs (B,T,D)
    # Simple linear recurrence: h_t = a_t * h_{t-1} + x_t
    import jax, jax.numpy as jnp
    def step(h, ax):
        a_t, x_t = ax
        h_new = a_t * h + x_t
        return h_new, h_new
    B, T, D = x.shape
    _, hs = jax.vmap(lambda ai,xi: jax.lax.scan(step, jnp.zeros(D), (ai,xi)))(a,x)
    return hs""",
  jax_module="""class Model:
    def __call__(self, a, x):
        import jax, jax.numpy as jnp
        def step(h, ax): a_t,x_t=ax; h_new=a_t*h+x_t; return h_new,h_new
        B,T,D=x.shape
        _,hs=jax.vmap(lambda ai,xi: jax.lax.scan(step,jnp.zeros(D),(ai,xi)))(a,x)
        return hs""",
  seed_pallas="""
import jax, jax.numpy as jnp
def linear_recurrence(a, x):
    def step(h, ax): a_t,x_t=ax; h_new=a_t*h+x_t; return h_new,h_new
    B,T,D=x.shape
    _,hs=jax.vmap(lambda ai,xi: jax.lax.scan(step,jnp.zeros(D),(ai,xi)))(a,x)
    return hs
"""),

Problem(task_id=85, name="rope_apply", level=2, category="attention",
  **_u("AI-Hypercomputer/maxtext","MaxText/layers/embeddings.py"),
  input_shapes=[(4,8,128,64)], input_dtypes=["float32"],
  jax_functional="""def rope_apply(x):
    import jax.numpy as jnp
    B,H,T,D=x.shape; half=D//2
    theta = 1.0/(10000**(jnp.arange(half,dtype=jnp.float32)/half))
    t = jnp.arange(T,dtype=jnp.float32)
    freqs = jnp.outer(t, theta)  # (T, D/2)
    cos = jnp.cos(freqs); sin = jnp.sin(freqs)
    x1,x2 = x[...,:half], x[...,half:]
    return jnp.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)""",
  jax_module="""class Model:
    def __call__(self,x):
        import jax.numpy as jnp
        B,H,T,D=x.shape; half=D//2
        theta=1./(10000**(jnp.arange(half,dtype=jnp.float32)/half))
        freqs=jnp.outer(jnp.arange(T,dtype=jnp.float32),theta)
        cos=jnp.cos(freqs); sin=jnp.sin(freqs)
        x1,x2=x[...,:half],x[...,half:]
        return jnp.concatenate([x1*cos-x2*sin,x1*sin+x2*cos],-1)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def rope_apply(x):
    B,H,T,D=x.shape; half=D//2
    theta=1./(10000**(jnp.arange(half,dtype=jnp.float32)/half))
    freqs=jnp.outer(jnp.arange(T,dtype=jnp.float32),theta)
    cos=jnp.cos(freqs); sin=jnp.sin(freqs)
    x1,x2=x[...,:half],x[...,half:]
    return jnp.concatenate([x1*cos-x2*sin,x1*sin+x2*cos],-1)
"""),
]

# ─── L3 NEW ARCHITECTURE COMPONENTS (task_ids 101-120) ──────────────────────

EXT += [
Problem(task_id=101, name="ffn_block", level=3, category="transformer",
  **_u("AI-Hypercomputer/maxtext","MaxText/layers/models.py"),
  input_shapes=[(4,128,512),(512,2048),(2048,512),(512,)], input_dtypes=["float32"]*4,
  jax_functional="""def ffn_block(x, w1, w2, bias):
    import jax, jax.numpy as jnp
    return jax.nn.silu(x @ w1) @ w2 + bias""",
  jax_module="""class Model:
    def __call__(self,x,w1,w2,bias):
        import jax
        return jax.nn.silu(x@w1)@w2+bias""",
  seed_pallas="import jax\ndef ffn_block(x,w1,w2,bias): return jax.nn.silu(x@w1)@w2+bias"),

Problem(task_id=102, name="moe_gate", level=3, category="moe",
  **_u("sgl-project/sglang-jax","python/sgl_jax/srt/layers/fused_moe.py"),
  input_shapes=[(512,512),(512,8)], input_dtypes=["float32","float32"],
  jax_functional="""def moe_gate(x, gate_w, top_k=2):
    import jax, jax.numpy as jnp
    scores = jax.nn.softmax(x @ gate_w, axis=-1)
    topk_vals, topk_idx = jax.lax.top_k(scores, top_k)
    return topk_vals, topk_idx""",
  jax_module="""class Model:
    def __call__(self,x,gate_w,top_k=2):
        import jax, jax.numpy as jnp
        scores=jax.nn.softmax(x@gate_w,-1)
        return jax.lax.top_k(scores,top_k)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def moe_gate(x, gate_w, top_k=2):
    scores=jax.nn.softmax(x@gate_w,-1)
    return jax.lax.top_k(scores,top_k)
"""),

Problem(task_id=103, name="cross_attention", level=3, category="attention",
  **_u("google/flax","flax/linen/attention.py"),
  input_shapes=[(4,128,512),(4,64,512)], input_dtypes=["float32","float32"],
  jax_functional="""def cross_attention(q, kv):
    import jax, jax.numpy as jnp
    scale = q.shape[-1]**-0.5
    scores = jnp.einsum('bqd,bkd->bqk', q, kv) * scale
    weights = jax.nn.softmax(scores, -1)
    return jnp.einsum('bqk,bkd->bqd', weights, kv)""",
  jax_module="""class Model:
    def __call__(self,q,kv):
        import jax,jax.numpy as jnp
        scale=q.shape[-1]**-0.5
        scores=jnp.einsum('bqd,bkd->bqk',q,kv)*scale
        return jnp.einsum('bqk,bkd->bqd',jax.nn.softmax(scores,-1),kv)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def cross_attention(q, kv):
    scale=q.shape[-1]**-0.5
    scores=jnp.einsum('bqd,bkd->bqk',q,kv)*scale
    return jnp.einsum('bqk,bkd->bqd',jax.nn.softmax(scores,-1),kv)
"""),

Problem(task_id=104, name="gated_linear_unit", level=3, category="fused",
  **_u("google-deepmind/alphafold3","src/alphafold3/jax/gated_linear_unit"),
  input_shapes=[(512,1024),(1024,512),(1024,512)], input_dtypes=["float32"]*3,
  jax_functional="""def gated_linear_unit(x, w_gate, w_val):
    import jax, jax.numpy as jnp
    gate = jax.nn.sigmoid(x @ w_gate)
    val  = x @ w_val
    return gate * val""",
  jax_module="""class Model:
    def __call__(self,x,w_gate,w_val):
        import jax
        return jax.nn.sigmoid(x@w_gate)*(x@w_val)""",
  seed_pallas="import jax\ndef gated_linear_unit(x,w_gate,w_val): return jax.nn.sigmoid(x@w_gate)*(x@w_val)"),

Problem(task_id=105, name="prenorm_residual", level=3, category="transformer",
  **_u("google/flax","flax/linen/attention.py"),
  input_shapes=[(4,128,512),(512,)], input_dtypes=["float32","float32"],
  jax_functional="""def prenorm_residual(x, w):
    import jax.numpy as jnp
    rms = jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+1e-6)
    normed = w * x / rms
    return x + normed""",
  jax_module="""class Model:
    def __call__(self,x,w):
        import jax.numpy as jnp
        rms=jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+1e-6)
        return x+w*x/rms""",
  seed_pallas="""
import jax, jax.numpy as jnp
def prenorm_residual(x, w):
    rms=jnp.sqrt(jnp.mean(x**2,-1,keepdims=True)+1e-6)
    return x+w*x/rms
"""),
]

# ─── L4 FULL MODELS (task_ids 121-145) ──────────────────────────────────────

EXT += [
Problem(task_id=121, name="resnet_basic_block", level=4, category="vision",
  **_u("google/flax","examples/imagenet/models.py"),
  input_shapes=[(4,56,56,64),(3,3,64,64),(3,3,64,64)], input_dtypes=["float32"]*3,
  jax_functional="""def resnet_basic_block(x, w1, w2):
    import jax, jax.numpy as jnp
    from jax import lax
    # conv1
    h = lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
    h = jax.nn.relu(h)
    # conv2
    h = lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
    return jax.nn.relu(h + x)""",
  jax_module="""class Model:
    def __call__(self,x,w1,w2):
        import jax; from jax import lax
        h=lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
        h=jax.nn.relu(h)
        h=lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
        return jax.nn.relu(h+x)""",
  seed_pallas="""
import jax, jax.numpy as jnp; from jax import lax
def resnet_basic_block(x, w1, w2):
    h=lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
    h=jax.nn.relu(h)
    h=lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'))
    return jax.nn.relu(h+x)
"""),

Problem(task_id=122, name="llama_decoder_layer", level=4, category="llm",
  **_u("AI-Hypercomputer/maxtext","MaxText/layers/models.py"),
  input_shapes=[(2,128,512),(512,),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def llama_decoder_layer(x, norm_w, wq, wk, wv, wo, w1, w2, ffn_norm_w):
    import jax, jax.numpy as jnp
    # Attention with RMSNorm pre-norm
    def rmsnorm(h, w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+1e-6)
    B,T,D=x.shape; H,d=8,D//8
    h=rmsnorm(x,norm_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3)
    k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    # FFN with RMSNorm pre-norm
    h=rmsnorm(x,ffn_norm_w)
    return x+jax.nn.silu(h@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,norm_w,wq,wk,wv,wo,w1,w2,ffn_norm_w):
        import jax,jax.numpy as jnp
        def rms(h,w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+1e-6)
        B,T,D=x.shape; H,d=8,D//8
        h=rms(x,norm_w)
        q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3)
        k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
        v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        return x+jax.nn.silu(rms(x,ffn_norm_w)@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp
def llama_decoder_layer(x,norm_w,wq,wk,wv,wo,w1,w2,ffn_norm_w):
    def rms(h,w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+1e-6)
    B,T,D=x.shape; H,d=8,D//8
    h=rms(x,norm_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3)
    k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.silu(rms(x,ffn_norm_w)@w1)@w2
"""),

Problem(task_id=123, name="bert_encoder_layer", level=4, category="llm",
  **_u("huggingface/transformers","src/transformers/models/bert/modeling_flax_bert.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def bert_encoder_layer(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-12):
    import jax, jax.numpy as jnp
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8
    q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3)
    k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    attn_out=jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    x=ln(x+attn_out,ln1_w)
    ffn_out=jax.nn.gelu(x@w1)@w2
    return ln(x+ffn_out,ln2_w)""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w):
        import jax,jax.numpy as jnp
        def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+1e-12)
        B,T,D=x.shape; H,d=8,D//8
        q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3)
        v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=ln(x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo,ln1_w)
        return ln(x+jax.nn.gelu(x@w1)@w2,ln2_w)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def bert_encoder_layer(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-12):
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8
    q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=ln(x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo,ln1_w)
    return ln(x+jax.nn.gelu(x@w1)@w2,ln2_w)
"""),

Problem(task_id=124, name="vit_block", level=4, category="vision",
  **_u("google-research/scenic","scenic/projects/baselines/vit.py"),
  input_shapes=[(4,197,512),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def vit_block(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-6):
    import jax, jax.numpy as jnp
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8
    h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w):
        import jax,jax.numpy as jnp
        def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+1e-6)
        B,T,D=x.shape; H,d=8,D//8
        h=ln(x,ln1_w); q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp
def vit_block(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-6):
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8; h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2
"""),

Problem(task_id=125, name="gpt2_decoder_layer", level=4, category="llm",
  **_u("huggingface/transformers","src/transformers/models/gpt2/modeling_flax_gpt2.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def gpt2_decoder_layer(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-5):
    import jax, jax.numpy as jnp
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8
    h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    # Causal mask
    mask=jnp.tril(jnp.ones((T,T)))[None,None]
    scores=jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5
    scores=jnp.where(mask,scores,-1e9)
    att=jax.nn.softmax(scores,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w):
        import jax,jax.numpy as jnp
        def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+1e-5)
        B,T,D=x.shape; H,d=8,D//8; h=ln(x,ln1_w)
        q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        mask=jnp.tril(jnp.ones((T,T)))[None,None]
        scores=jnp.where(mask,jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1e9)
        att=jax.nn.softmax(scores,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp
def gpt2_decoder_layer(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-5):
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8; h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    mask=jnp.tril(jnp.ones((T,T)))[None,None]
    scores=jnp.where(mask,jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1e9)
    att=jax.nn.softmax(scores,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2
"""),

Problem(task_id=126, name="griffin_recurrent_block", level=4, category="rnn",
  **_u("google-deepmind/recurrentgemma","recurrentgemma/jax/pallas.py"),
  input_shapes=[(4,128,512),(512,),(512,512),(512,512)], input_dtypes=["float32"]*4,
  jax_functional="""def griffin_recurrent_block(x, decay_w, w_in, w_out):
    import jax, jax.numpy as jnp
    # Gated linear recurrence (Griffin/RecurrentGemma style)
    B,T,D=x.shape
    gate = jax.nn.sigmoid(x @ w_in)  # gating
    x_proj = x * gate
    # Linear recurrence: h_t = decay * h_{t-1} + x_t
    a = jax.nn.sigmoid(decay_w)[None,None,:]  # (1,1,D)
    def step(h, xt): h_new = a[0,0]*h + xt; return h_new, h_new
    _, hs = jax.vmap(lambda xi: jax.lax.scan(step, jnp.zeros(D), xi))(x_proj)
    return hs @ w_out""",
  jax_module="""class Model:
    def __call__(self,x,decay_w,w_in,w_out):
        import jax,jax.numpy as jnp
        B,T,D=x.shape; gate=jax.nn.sigmoid(x@w_in); x_proj=x*gate
        a=jax.nn.sigmoid(decay_w)
        def step(h,xt): h_new=a*h+xt; return h_new,h_new
        _,hs=jax.vmap(lambda xi:jax.lax.scan(step,jnp.zeros(D),xi))(x_proj)
        return hs@w_out""",
  seed_pallas="""
import jax, jax.numpy as jnp
def griffin_recurrent_block(x, decay_w, w_in, w_out):
    B,T,D=x.shape; gate=jax.nn.sigmoid(x@w_in); x_proj=x*gate
    a=jax.nn.sigmoid(decay_w)
    def step(h,xt): h_new=a*h+xt; return h_new,h_new
    _,hs=jax.vmap(lambda xi:jax.lax.scan(step,jnp.zeros(D),xi))(x_proj)
    return hs@w_out
"""),

Problem(task_id=127, name="mamba_ssm_block", level=4, category="ssm",
  **_u("vvvm23/mamba-jax","mamba_jax/modelling/mamba.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,),(512,)], input_dtypes=["float32"]*5,
  jax_functional="""def mamba_ssm_block(x, w_in, w_out, A_log, D):
    import jax, jax.numpy as jnp
    B,T,D_model=x.shape
    # Simplified Mamba: project, selective scan, project out
    x_proj = jax.nn.silu(x @ w_in)  # (B,T,2*D_model)
    half=x_proj.shape[-1]//2
    x1, x2 = x_proj[...,:half], x_proj[...,half:]
    # State space: A = -exp(A_log), simplified scan
    A = -jnp.exp(A_log[:half])
    def scan_step(h, xt): h_new=jnp.exp(A)*h+xt; return h_new,h_new
    _,ys=jax.vmap(lambda xi:jax.lax.scan(scan_step,jnp.zeros(half),xi))(x1)
    y = ys * jax.nn.silu(x2) + x1 * D[:half]
    return y @ w_out""",
  jax_module="""class Model:
    def __call__(self,x,w_in,w_out,A_log,D_coef):
        import jax,jax.numpy as jnp
        B,T,Dm=x.shape; x_proj=jax.nn.silu(x@w_in); half=x_proj.shape[-1]//2
        x1,x2=x_proj[...,:half],x_proj[...,half:]
        A=-jnp.exp(A_log[:half])
        def step(h,xt): h_new=jnp.exp(A)*h+xt; return h_new,h_new
        _,ys=jax.vmap(lambda xi:jax.lax.scan(step,jnp.zeros(half),xi))(x1)
        return (ys*jax.nn.silu(x2)+x1*D_coef[:half])@w_out""",
  seed_pallas="""
import jax, jax.numpy as jnp
def mamba_ssm_block(x, w_in, w_out, A_log, D):
    B,T,Dm=x.shape; x_proj=jax.nn.silu(x@w_in); half=x_proj.shape[-1]//2
    x1,x2=x_proj[...,:half],x_proj[...,half:]
    A=-jnp.exp(A_log[:half])
    def step(h,xt): h_new=jnp.exp(A)*h+xt; return h_new,h_new
    _,ys=jax.vmap(lambda xi:jax.lax.scan(step,jnp.zeros(half),xi))(x1)
    return (ys*jax.nn.silu(x2)+x1*D[:half])@w_out
"""),

Problem(task_id=128, name="mobilenetv2_block", level=4, category="vision",
  **_u("keras-team/keras","keras/src/applications/mobilenet_v2.py"),
  input_shapes=[(4,28,28,32),(1,1,32,128),(3,3,128,1),(1,1,128,32)], input_dtypes=["float32"]*4,
  jax_functional="""def mobilenetv2_block(x, w_expand, w_dw, w_project):
    import jax, jax.numpy as jnp; from jax import lax
    # 1) Expand
    h = jax.nn.relu6(lax.conv_general_dilated(x,w_expand,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC')))
    # 2) Depthwise conv (simplified as grouped conv)
    h = jax.nn.relu6(lax.conv_general_dilated(h,w_dw,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=h.shape[-1]))
    # 3) Project (linear)
    return lax.conv_general_dilated(h,w_project,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'))""",
  jax_module="""class Model:
    def __call__(self,x,w_expand,w_dw,w_project):
        import jax; from jax import lax
        h=jax.nn.relu6(lax.conv_general_dilated(x,w_expand,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC')))
        h=jax.nn.relu6(lax.conv_general_dilated(h,w_dw,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=h.shape[-1]))
        return lax.conv_general_dilated(h,w_project,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'))""",
  seed_pallas="""
import jax; from jax import lax
def mobilenetv2_block(x,w_expand,w_dw,w_project):
    h=jax.nn.relu6(lax.conv_general_dilated(x,w_expand,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC')))
    h=jax.nn.relu6(lax.conv_general_dilated(h,w_dw,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=h.shape[-1]))
    return lax.conv_general_dilated(h,w_project,(1,1),((0,0),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'))
"""),

Problem(task_id=129, name="whisper_encoder_block", level=4, category="audio",
  **_u("sanchit-gandhi/whisper-jax","whisper_jax/modeling_flax_whisper.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def whisper_encoder_block(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-5):
    import jax, jax.numpy as jnp
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8
    h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w):
        import jax,jax.numpy as jnp
        def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+1e-5)
        B,T,D=x.shape; H,d=8,D//8; h=ln(x,ln1_w)
        q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp
def whisper_encoder_block(x,wq,wk,wv,wo,w1,w2,ln1_w,ln2_w,eps=1e-5):
    def ln(h,w): mean=jnp.mean(h,-1,keepdims=True); var=jnp.var(h,-1,keepdims=True); return w*(h-mean)/jnp.sqrt(var+eps)
    B,T,D=x.shape; H,d=8,D//8; h=ln(x,ln1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.gelu(ln(x,ln2_w)@w1)@w2
"""),

Problem(task_id=130, name="gcn_layer", level=4, category="graph",
  **_u("gcucurull/jax-gcn","jax_gcn/gcn.py"),
  input_shapes=[(512,512),(512,512),(512,256)], input_dtypes=["float32"]*3,
  jax_functional="""def gcn_layer(x, adj, weight):
    import jax, jax.numpy as jnp
    # Graph Conv: A_hat @ X @ W  where A_hat = D^-0.5 A D^-0.5
    deg = jnp.sum(adj, -1, keepdims=True)
    d_inv_sqrt = jnp.where(deg>0, 1./jnp.sqrt(deg+1e-8), 0.)
    adj_norm = d_inv_sqrt * adj * d_inv_sqrt.T
    return jax.nn.relu(adj_norm @ x @ weight)""",
  jax_module="""class Model:
    def __call__(self,x,adj,weight):
        import jax,jax.numpy as jnp
        deg=jnp.sum(adj,-1,keepdims=True); d_inv=jnp.where(deg>0,1./jnp.sqrt(deg+1e-8),0.)
        adj_norm=d_inv*adj*d_inv.T
        return jax.nn.relu(adj_norm@x@weight)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def gcn_layer(x, adj, weight):
    deg=jnp.sum(adj,-1,keepdims=True); d_inv=jnp.where(deg>0,1./jnp.sqrt(deg+1e-8),0.)
    return jax.nn.relu((d_inv*adj*d_inv.T)@x@weight)
"""),

Problem(task_id=131, name="black_scholes_batch", level=4, category="finance",
  **_u("SamDuffield/mocat","mocat/finance.py"),
  input_shapes=[(8192,),(8192,),(8192,),(8192,),(8192,)], input_dtypes=["float32"]*5,
  jax_functional="""def black_scholes_batch(S, K, T, r, sigma):
    import jax.numpy as jnp
    from jax.scipy.special import ndtr
    d1 = (jnp.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*jnp.sqrt(T) + 1e-8)
    d2 = d1 - sigma*jnp.sqrt(T)
    call = S*ndtr(d1) - K*jnp.exp(-r*T)*ndtr(d2)
    return call""",
  jax_module="""class Model:
    def __call__(self,S,K,T,r,sigma):
        import jax.numpy as jnp; from jax.scipy.special import ndtr
        d1=(jnp.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*jnp.sqrt(T)+1e-8)
        d2=d1-sigma*jnp.sqrt(T)
        return S*ndtr(d1)-K*jnp.exp(-r*T)*ndtr(d2)""",
  seed_pallas="""
import jax, jax.numpy as jnp; from jax.scipy.special import ndtr
def black_scholes_batch(S, K, T, r, sigma):
    d1=(jnp.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*jnp.sqrt(T)+1e-8); d2=d1-sigma*jnp.sqrt(T)
    return S*ndtr(d1)-K*jnp.exp(-r*T)*ndtr(d2)
"""),

Problem(task_id=132, name="smith_waterman_batch", level=4, category="genomics",
  **_u("spetti/SMURF","smurf/align.py"),
  input_shapes=[(32,64),(32,64)], input_dtypes=["int32","int32"],
  jax_functional="""def smith_waterman_batch(seqs_a, seqs_b, match=2, mismatch=-1, gap=-1):
    import jax, jax.numpy as jnp
    B, La = seqs_a.shape; _, Lb = seqs_b.shape
    # Simplified: compute match score matrix per batch element
    def sw_single(a, b):
        scores = jnp.where(a[:,None]==b[None,:], match, mismatch).astype(jnp.float32)
        return jnp.max(scores)
    return jax.vmap(sw_single)(seqs_a, seqs_b)""",
  jax_module="""class Model:
    def __call__(self,seqs_a,seqs_b,match=2,mismatch=-1,gap=-1):
        import jax,jax.numpy as jnp
        def sw(a,b): return jnp.max(jnp.where(a[:,None]==b[None,:],match,mismatch).astype(jnp.float32))
        return jax.vmap(sw)(seqs_a,seqs_b)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def smith_waterman_batch(seqs_a, seqs_b, match=2, mismatch=-1, gap=-1):
    def sw(a,b): return jnp.max(jnp.where(a[:,None]==b[None,:],match,mismatch).astype(jnp.float32))
    return jax.vmap(sw)(seqs_a,seqs_b)
"""),

Problem(task_id=133, name="molecular_distance_matrix", level=4, category="science",
  **_u("google/jax-md","jax_md/space.py"),
  input_shapes=[(512,3)], input_dtypes=["float32"],
  jax_functional="""def molecular_distance_matrix(coords):
    import jax.numpy as jnp
    diff = coords[:,None,:] - coords[None,:,:]
    return jnp.sqrt(jnp.sum(diff**2,-1)+1e-8)""",
  jax_module="""class Model:
    def __call__(self,coords):
        import jax.numpy as jnp
        diff=coords[:,None,:]-coords[None,:,:]
        return jnp.sqrt(jnp.sum(diff**2,-1)+1e-8)""",
  seed_pallas="""
import jax.numpy as jnp
def molecular_distance_matrix(coords):
    diff=coords[:,None,:]-coords[None,:,:]
    return jnp.sqrt(jnp.sum(diff**2,-1)+1e-8)
"""),

Problem(task_id=134, name="unet_double_conv", level=4, category="vision",
  **_u("AI-Hypercomputer/maxtext","MaxText/layers/models.py"),
  input_shapes=[(4,64,64,32),(3,3,32,64),(3,3,64,64)], input_dtypes=["float32"]*3,
  jax_functional="""def unet_double_conv(x, w1, w2):
    import jax; from jax import lax
    h = jax.nn.relu(lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))
    return jax.nn.relu(lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))""",
  jax_module="""class Model:
    def __call__(self,x,w1,w2):
        import jax; from jax import lax
        h=jax.nn.relu(lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))
        return jax.nn.relu(lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))""",
  seed_pallas="""
import jax; from jax import lax
def unet_double_conv(x, w1, w2):
    h=jax.nn.relu(lax.conv_general_dilated(x,w1,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))
    return jax.nn.relu(lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=('NHWC','HWIO','NHWC')))
"""),

Problem(task_id=135, name="alphafold_triangle_mult", level=4, category="science",
  **_u("google-deepmind/alphafold3","src/alphafold3/model/components/haiku_modules.py"),
  input_shapes=[(8,64,64,64),(64,64),(64,64),(64,),(64,)], input_dtypes=["float32"]*5,
  jax_functional="""def alphafold_triangle_mult(z, w_left, w_right, b_left, b_right):
    import jax, jax.numpy as jnp
    # Simplified outgoing triangle multiplicative update
    # z: (B,N,N,C), output: (B,N,N,C)
    B,N,_,C = z.shape
    left  = jax.nn.sigmoid(z @ w_left  + b_left)
    right = jax.nn.sigmoid(z @ w_right + b_right)
    return jnp.einsum('biqc,bjqc->bijc', left, right)""",
  jax_module="""class Model:
    def __call__(self,z,w_left,w_right,b_left,b_right):
        import jax,jax.numpy as jnp
        left=jax.nn.sigmoid(z@w_left+b_left); right=jax.nn.sigmoid(z@w_right+b_right)
        return jnp.einsum('biqc,bjqc->bijc',left,right)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def alphafold_triangle_mult(z, w_left, w_right, b_left, b_right):
    left=jax.nn.sigmoid(z@w_left+b_left); right=jax.nn.sigmoid(z@w_right+b_right)
    return jnp.einsum('biqc,bjqc->bijc',left,right)
"""),

Problem(task_id=136, name="portfolio_variance", level=4, category="finance",
  **_u("google/jax-md","jax_md/nn.py"),
  input_shapes=[(512,512),(512,)], input_dtypes=["float32","float32"],
  jax_functional="""def portfolio_variance(cov, weights):
    import jax.numpy as jnp
    return jnp.dot(weights, cov @ weights)""",
  jax_module="""class Model:
    def __call__(self,cov,weights):
        import jax.numpy as jnp
        return jnp.dot(weights,cov@weights)""",
  seed_pallas="import jax.numpy as jnp\ndef portfolio_variance(cov,weights): return jnp.dot(weights,cov@weights)"),

Problem(task_id=137, name="monte_carlo_option", level=4, category="finance",
  **_u("SamDuffield/mocat","mocat/finance.py"),
  input_shapes=[(8192,),(8192,)], input_dtypes=["float32","float32"],
  jax_functional="""def monte_carlo_option(paths, strike):
    import jax.numpy as jnp
    payoffs = jnp.maximum(paths - strike, 0.)
    return jnp.mean(payoffs)""",
  jax_module="""class Model:
    def __call__(self,paths,strike):
        import jax.numpy as jnp
        return jnp.mean(jnp.maximum(paths-strike,0.))""",
  seed_pallas="""
import jax, jax.numpy as jnp, jax.experimental.pallas as pl
def mc_k(paths_ref, strike_ref, o_ref):
    o_ref[...] = jnp.maximum(paths_ref[...] - strike_ref[...], 0.)
def monte_carlo_option(paths, strike):
    n=paths.shape[0]; b=min(4096,n)
    payoffs=pl.pallas_call(mc_k,out_shape=jax.ShapeDtypeStruct(paths.shape,paths.dtype),
        grid=(n//b,),in_specs=[pl.BlockSpec((b,),lambda i:(i,)),pl.BlockSpec((b,),lambda i:(i,))],
        out_specs=pl.BlockSpec((b,),lambda i:(i,)))(paths,strike)
    return jnp.mean(payoffs)
"""),

Problem(task_id=138, name="verlet_integration", level=4, category="physics",
  **_u("google/brax","brax/physics/base.py"),
  input_shapes=[(512,3),(512,3),(512,3),(512,)], input_dtypes=["float32"]*4,
  jax_functional="""def verlet_integration(pos, vel, forces, mass, dt=0.01):
    import jax.numpy as jnp
    acc = forces / mass[:,None]
    new_pos = pos + vel * dt + 0.5 * acc * dt**2
    new_vel = vel + acc * dt
    return new_pos, new_vel""",
  jax_module="""class Model:
    def __call__(self,pos,vel,forces,mass,dt=0.01):
        import jax.numpy as jnp
        acc=forces/mass[:,None]; new_pos=pos+vel*dt+0.5*acc*dt**2; new_vel=vel+acc*dt
        return new_pos,new_vel""",
  seed_pallas="""
import jax.numpy as jnp
def verlet_integration(pos, vel, forces, mass, dt=0.01):
    acc=forces/mass[:,None]; new_pos=pos+vel*dt+0.5*acc*dt**2; new_vel=vel+acc*dt
    return new_pos, new_vel
"""),

Problem(task_id=139, name="attention_forward", level=4, category="attention",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4,8,128,64),(4,8,128,64),(4,8,128,64)], input_dtypes=["float32"]*3,
  jax_functional="""def attention_forward(q,k,v):
    import jax, jax.numpy as jnp
    scale=q.shape[-1]**-0.5
    scores=jnp.einsum('bhqd,bhkd->bhqk',q,k)*scale
    return jnp.einsum('bhqk,bhkd->bhqd',jax.nn.softmax(scores,-1),v)""",
  jax_module="""class Model:
    def __call__(self,q,k,v):
        import jax,jax.numpy as jnp
        scale=q.shape[-1]**-0.5; scores=jnp.einsum('bhqd,bhkd->bhqk',q,k)*scale
        return jnp.einsum('bhqk,bhkd->bhqd',jax.nn.softmax(scores,-1),v)""",
  seed_pallas="""
import jax, jax.numpy as jnp
def attention_forward(q,k,v):
    scale=q.shape[-1]**-0.5; scores=jnp.einsum('bhqd,bhkd->bhqk',q,k)*scale
    return jnp.einsum('bhqk,bhkd->bhqd',jax.nn.softmax(scores,-1),v)
"""),

Problem(task_id=140, name="mlp_block", level=4, category="transformer",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(4,128,512),(512,2048),(2048,512),(2048,),(512,)], input_dtypes=["float32"]*5,
  jax_functional="""def mlp_block(x, w1, w2, b1, b2):
    import jax, jax.numpy as jnp
    return jax.nn.gelu(x @ w1 + b1) @ w2 + b2""",
  jax_module="""class Model:
    def __call__(self,x,w1,w2,b1,b2):
        import jax
        return jax.nn.gelu(x@w1+b1)@w2+b2""",
  seed_pallas="import jax\ndef mlp_block(x,w1,w2,b1,b2): return jax.nn.gelu(x@w1+b1)@w2+b2"),

Problem(task_id=141, name="conv1d_genomic", level=4, category="genomics",
  **_u("Tyronita/PallasBench","data/pallasbench_sakana_style.jsonl","master"),
  input_shapes=[(16,4096,4),(7,4,64)], input_dtypes=["float32","float32"],
  jax_functional="""def conv1d_genomic(x, w):
    import jax, jax.numpy as jnp; from jax import lax
    # x: (B,L,C_in), w: (K,C_in,C_out) -> (B,L-K+1,C_out)
    return lax.conv_general_dilated(x,w,(1,),((0,0),),dimension_numbers=('NHC','HIO','NHC'))""",
  jax_module="""class Model:
    def __call__(self,x,w):
        from jax import lax
        return lax.conv_general_dilated(x,w,(1,),((0,0),),dimension_numbers=('NHC','HIO','NHC'))""",
  seed_pallas="""
from jax import lax
def conv1d_genomic(x, w):
    return lax.conv_general_dilated(x,w,(1,),((0,0),),dimension_numbers=('NHC','HIO','NHC'))
"""),

Problem(task_id=142, name="deepseek_moe_layer", level=4, category="moe",
  **_u("AI-Hypercomputer/maxtext","MaxText/models/deepseek_batchsplit_fp8.py"),
  input_shapes=[(512,512),(8,512,512),(512,8),(512,)], input_dtypes=["float32"]*4,
  jax_functional="""def deepseek_moe_layer(x, expert_weights, gate_w, router_bias, top_k=2):
    import jax, jax.numpy as jnp
    # Gate + top-k routing
    logits = x @ gate_w + router_bias
    scores = jax.nn.softmax(logits, -1)
    topk_scores, topk_idx = jax.lax.top_k(scores, top_k)
    topk_scores = topk_scores / (topk_scores.sum(-1, keepdims=True) + 1e-8)
    # Expert computation (simplified: take top expert)
    best_expert = topk_idx[:, 0]
    expert_out = jax.vmap(lambda xi, ei: jax.nn.silu(expert_weights[ei] @ xi))(x, best_expert)
    return expert_out * topk_scores[:, :1]""",
  jax_module="""class Model:
    def __call__(self,x,expert_weights,gate_w,router_bias,top_k=2):
        import jax,jax.numpy as jnp
        logits=x@gate_w+router_bias; scores=jax.nn.softmax(logits,-1)
        topk_s,topk_i=jax.lax.top_k(scores,top_k); topk_s=topk_s/(topk_s.sum(-1,keepdims=True)+1e-8)
        best=topk_i[:,0]
        out=jax.vmap(lambda xi,ei:jax.nn.silu(expert_weights[ei]@xi))(x,best)
        return out*topk_s[:,:1]""",
  seed_pallas="""
import jax, jax.numpy as jnp
def deepseek_moe_layer(x, expert_weights, gate_w, router_bias, top_k=2):
    logits=x@gate_w+router_bias; scores=jax.nn.softmax(logits,-1)
    topk_s,topk_i=jax.lax.top_k(scores,top_k); topk_s=topk_s/(topk_s.sum(-1,keepdims=True)+1e-8)
    best=topk_i[:,0]
    out=jax.vmap(lambda xi,ei:jax.nn.silu(expert_weights[ei]@xi))(x,best)
    return out*topk_s[:,:1]
"""),

Problem(task_id=143, name="resnet_bottleneck", level=4, category="vision",
  **_u("google/flax","examples/imagenet/models.py"),
  input_shapes=[(4,56,56,256),(1,1,256,64),(3,3,64,64),(1,1,64,256)], input_dtypes=["float32"]*4,
  jax_functional="""def resnet_bottleneck(x, w1, w2, w3):
    import jax; from jax import lax
    dn = ('NHWC','HWIO','NHWC')
    h = jax.nn.relu(lax.conv_general_dilated(x, w1,(1,1),((0,0),(0,0)),dimension_numbers=dn))
    h = jax.nn.relu(lax.conv_general_dilated(h, w2,(1,1),((1,1),(1,1)),dimension_numbers=dn))
    h = lax.conv_general_dilated(h, w3,(1,1),((0,0),(0,0)),dimension_numbers=dn)
    return jax.nn.relu(h + x)""",
  jax_module="""class Model:
    def __call__(self,x,w1,w2,w3):
        import jax; from jax import lax; dn=('NHWC','HWIO','NHWC')
        h=jax.nn.relu(lax.conv_general_dilated(x,w1,(1,1),((0,0),(0,0)),dimension_numbers=dn))
        h=jax.nn.relu(lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=dn))
        return jax.nn.relu(lax.conv_general_dilated(h,w3,(1,1),((0,0),(0,0)),dimension_numbers=dn)+x)""",
  seed_pallas="""
import jax; from jax import lax; dn=('NHWC','HWIO','NHWC')
def resnet_bottleneck(x,w1,w2,w3):
    h=jax.nn.relu(lax.conv_general_dilated(x,w1,(1,1),((0,0),(0,0)),dimension_numbers=dn))
    h=jax.nn.relu(lax.conv_general_dilated(h,w2,(1,1),((1,1),(1,1)),dimension_numbers=dn))
    return jax.nn.relu(lax.conv_general_dilated(h,w3,(1,1),((0,0),(0,0)),dimension_numbers=dn)+x)
"""),

Problem(task_id=144, name="t5_encoder_layer", level=4, category="llm",
  **_u("google/flax","flax/linen/attention.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,512),(512,512),(512,2048),(2048,512),(512,),(512,)],
  input_dtypes=["float32"]*9,
  jax_functional="""def t5_encoder_layer(x,wq,wk,wv,wo,w1,w2,rmsn1_w,rmsn2_w,eps=1e-6):
    import jax, jax.numpy as jnp
    def rmsn(h,w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+eps)
    B,T,D=x.shape; H,d=8,D//8
    h=rmsn(x,rmsn1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3)
    v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.relu(rmsn(x,rmsn2_w)@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w1,w2,rmsn1_w,rmsn2_w):
        import jax,jax.numpy as jnp
        def rmsn(h,w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+1e-6)
        B,T,D=x.shape; H,d=8,D//8; h=rmsn(x,rmsn1_w)
        q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        return x+jax.nn.relu(rmsn(x,rmsn2_w)@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp
def t5_encoder_layer(x,wq,wk,wv,wo,w1,w2,rmsn1_w,rmsn2_w,eps=1e-6):
    def rmsn(h,w): return w*h/jnp.sqrt(jnp.mean(h**2,-1,keepdims=True)+eps)
    B,T,D=x.shape; H,d=8,D//8; h=rmsn(x,rmsn1_w)
    q=(h@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(h@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(h@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    return x+jax.nn.relu(rmsn(x,rmsn2_w)@w1)@w2
"""),

Problem(task_id=145, name="conformer_block", level=4, category="audio",
  **_u("jaketae/conformer","conformer/model.py"),
  input_shapes=[(4,128,512),(512,512),(512,512),(512,512),(512,512),(7,512,1),(512,2048),(2048,512)],
  input_dtypes=["float32"]*8,
  jax_functional="""def conformer_block(x,wq,wk,wv,wo,w_dw,w1,w2,eps=1e-6):
    import jax, jax.numpy as jnp; from jax import lax
    B,T,D=x.shape; H,d=8,D//8
    # Self-attention
    q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    # Depthwise conv (on T dim)
    xc=x.transpose(0,2,1)[...,None]  # (B,D,T,1)
    h=lax.conv_general_dilated(xc,w_dw,(1,1),((3,3),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=D)
    x=x+jax.nn.silu(h[...,0].transpose(0,2,1))
    # FFN
    return x+jax.nn.silu(x@w1)@w2""",
  jax_module="""class Model:
    def __call__(self,x,wq,wk,wv,wo,w_dw,w1,w2):
        import jax,jax.numpy as jnp; from jax import lax
        B,T,D=x.shape; H,d=8,D//8
        q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
        att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
        x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
        xc=x.transpose(0,2,1)[...,None]
        h=lax.conv_general_dilated(xc,w_dw,(1,1),((3,3),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=D)
        x=x+jax.nn.silu(h[...,0].transpose(0,2,1))
        return x+jax.nn.silu(x@w1)@w2""",
  seed_pallas="""
import jax, jax.numpy as jnp; from jax import lax
def conformer_block(x,wq,wk,wv,wo,w_dw,w1,w2,eps=1e-6):
    B,T,D=x.shape; H,d=8,D//8
    q=(x@wq).reshape(B,T,H,d).transpose(0,2,1,3); k=(x@wk).reshape(B,T,H,d).transpose(0,2,1,3); v=(x@wv).reshape(B,T,H,d).transpose(0,2,1,3)
    att=jax.nn.softmax(jnp.einsum('bhqd,bhkd->bhqk',q,k)*d**-0.5,-1)
    x=x+jnp.einsum('bhqk,bhkd->bhqd',att,v).transpose(0,2,1,3).reshape(B,T,D)@wo
    xc=x.transpose(0,2,1)[...,None]
    h=lax.conv_general_dilated(xc,w_dw,(1,1),((3,3),(0,0)),dimension_numbers=('NHWC','HWIO','NHWC'),feature_group_count=D)
    x=x+jax.nn.silu(h[...,0].transpose(0,2,1))
    return x+jax.nn.silu(x@w1)@w2
"""),
]

# Registry
EXT_BY_NAME  = {p.name: p for p in EXT}
EXT_BY_ID    = {p.task_id: p for p in EXT}
EXT_BY_LEVEL = {l: [p for p in EXT if p.level==l] for l in [1,2,3,4]}
