"""Pure JAX reference implementations for PallasBench tasks.

Each function uses only jax.numpy / jax.lax / jax.nn — no Pallas.
These serve as correctness references and performance baselines.
"""

from functools import partial

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Level 1: Activation
# ---------------------------------------------------------------------------

@jax.jit
def jax_relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(x, 0)


@jax.jit
def jax_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x)


@jax.jit
def jax_silu(x: jax.Array) -> jax.Array:
    return jax.nn.silu(x)


@jax.jit
def jax_sigmoid(x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(x)


@jax.jit
def jax_tanh(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


# ---------------------------------------------------------------------------
# Level 1: Normalization
# ---------------------------------------------------------------------------

@jax.jit
def jax_layernorm(x: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)


@jax.jit
def jax_rmsnorm(x: jax.Array) -> jax.Array:
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)


# ---------------------------------------------------------------------------
# Level 1: MatMul
# ---------------------------------------------------------------------------

@jax.jit
def jax_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    return x @ y


@jax.jit
def jax_batched_matmul(x: jax.Array, y: jax.Array) -> jax.Array:
    return x @ y


@jax.jit
def jax_outer_product(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.outer(x, y)


# ---------------------------------------------------------------------------
# Level 1: Reduce
# ---------------------------------------------------------------------------

@jax.jit
def jax_reduce_sum(x: jax.Array) -> jax.Array:
    return jnp.sum(x, axis=-1)


@jax.jit
def jax_reduce_max(x: jax.Array) -> jax.Array:
    return jnp.max(x, axis=-1)


@jax.jit
def jax_reduce_mean(x: jax.Array) -> jax.Array:
    return jnp.mean(x, axis=-1)


# ---------------------------------------------------------------------------
# Level 1: Softmax
# ---------------------------------------------------------------------------

@jax.jit
def jax_softmax(x: jax.Array) -> jax.Array:
    return jax.nn.softmax(x, axis=-1)


@jax.jit
def jax_log_softmax(x: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(x, axis=-1)


# ---------------------------------------------------------------------------
# Level 1: Elementwise
# ---------------------------------------------------------------------------

@jax.jit
def jax_exp(x: jax.Array) -> jax.Array:
    return jnp.exp(x)


@jax.jit
def jax_log(x: jax.Array) -> jax.Array:
    return jnp.log(x + 1e-7)


@jax.jit
def jax_add(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y


@jax.jit
def jax_multiply(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * y


@jax.jit
def jax_rsqrt(x: jax.Array) -> jax.Array:
    return jax.lax.rsqrt(x + 1e-5)


@jax.jit
def jax_clamp(x: jax.Array) -> jax.Array:
    return jnp.clip(x, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Level 1: Loss
# ---------------------------------------------------------------------------

@jax.jit
def jax_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels * log_probs, axis=-1)


@jax.jit
def jax_mse_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
    diff = pred - target
    return jnp.mean(diff * diff, axis=-1)


@jax.jit
def jax_cosine_sim(x: jax.Array, y: jax.Array) -> jax.Array:
    dot = jnp.sum(x * y, axis=-1)
    norm_x = jnp.sqrt(jnp.sum(x * x, axis=-1))
    norm_y = jnp.sqrt(jnp.sum(y * y, axis=-1))
    return dot / (norm_x * norm_y + 1e-8)


# ---------------------------------------------------------------------------
# Level 1: Index
# ---------------------------------------------------------------------------

@jax.jit
def jax_embedding_lookup(table: jax.Array, indices: jax.Array) -> jax.Array:
    return table[indices]


@jax.jit
def jax_one_hot(indices: jax.Array) -> jax.Array:
    return jax.nn.one_hot(indices, 1024, dtype=jnp.float32)


@jax.jit
def jax_nucleotide_onehot(seq: jax.Array) -> jax.Array:
    return jax.nn.one_hot(seq, 4, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Level 2: Fusion patterns
# ---------------------------------------------------------------------------

@jax.jit
def jax_matmul_relu(x: jax.Array, w: jax.Array) -> jax.Array:
    return jnp.maximum(x @ w, 0)


@jax.jit
def jax_matmul_gelu(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.nn.gelu(x @ w)


@jax.jit
def jax_matmul_silu(x: jax.Array, w: jax.Array) -> jax.Array:
    return jax.nn.silu(x @ w)


@jax.jit
def jax_rmsnorm_residual(x: jax.Array, residual: jax.Array) -> jax.Array:
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    normed = x / jnp.sqrt(ms + 1e-5)
    return normed + residual


@jax.jit
def jax_layernorm_residual(x: jax.Array, residual: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normed = (x - mean) / jnp.sqrt(var + 1e-5)
    return normed + residual


@jax.jit
def jax_swiglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    return gate * up


@jax.jit
def jax_geglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array) -> jax.Array:
    gate = jax.nn.gelu(x @ w_gate)
    up = x @ w_up
    return gate * up


@jax.jit
def jax_linear_bias_relu(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.maximum(x @ w + b, 0)


@jax.jit
def jax_qk_softmax(q: jax.Array, k: jax.Array) -> jax.Array:
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(d_k)
    return jax.nn.softmax(scores, axis=-1)


@jax.jit
def jax_fused_softmax_cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels * log_probs, axis=-1)


@jax.jit
def jax_sigmoid_bce(logits: jax.Array, targets: jax.Array) -> jax.Array:
    max_val = jnp.maximum(-logits, 0.0)
    loss = max_val + jnp.log(jnp.exp(-max_val) + jnp.exp(-logits - max_val))
    return loss - targets * logits + targets * loss


@jax.jit
def jax_pwm_scan(seq_onehot: jax.Array, pwm: jax.Array) -> jax.Array:
    motif_len = pwm.shape[1]
    seq_len = seq_onehot.shape[0]
    out_len = seq_len - motif_len + 1
    scores = jnp.zeros(out_len, dtype=seq_onehot.dtype)
    for pos in range(motif_len):
        scores = scores + jnp.sum(seq_onehot[pos:pos + out_len, :] * pwm[:, pos][None, :], axis=-1)
    return scores


@jax.jit
def jax_pairwise_distance(x: jax.Array) -> jax.Array:
    diff = x[:, None, :] - x[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-8)


# ---------------------------------------------------------------------------
# Level 3: Architecture components
# ---------------------------------------------------------------------------

@jax.jit
def jax_flash_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v


@jax.jit
def jax_multi_head_attention(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> jax.Array:
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-2, -1) / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v


@jax.jit
def jax_gated_mlp(
    x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array
) -> jax.Array:
    gate = jax.nn.silu(x @ w_gate)
    up = x @ w_up
    hidden = gate * up
    return hidden @ w_down


@jax.jit
def jax_triangle_update(pair: jax.Array, mask: jax.Array) -> jax.Array:
    n, _, c = pair.shape
    left_proj = pair * mask[:, :, None]
    right_proj = pair * mask[:, :, None]
    left_t = left_proj.transpose(2, 0, 1)
    right_t = right_proj.transpose(2, 0, 1)
    update = jnp.sum(left_t[:, :, :, None] * right_t[:, None, :, :], axis=2)
    update = update.transpose(1, 2, 0)
    return pair + update


@jax.jit
def jax_transformer_block(
    x: jax.Array, wq: jax.Array, wk: jax.Array,
    wv: jax.Array, wo: jax.Array, w_ff: jax.Array
) -> jax.Array:
    d_head = wq.shape[-1]
    # Pre-norm
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(ms + 1e-5)
    # Attention
    q = x_norm @ wq
    k = x_norm @ wk
    v = x_norm @ wv
    scores = q @ k.T / jnp.sqrt(jnp.float32(d_head))
    weights = jax.nn.softmax(scores, axis=-1)
    attn_out = weights @ v
    attn_proj = attn_out @ wo
    # Residual + MLP
    h = x + attn_proj
    ms2 = jnp.mean(h ** 2, axis=-1, keepdims=True)
    h_norm = h * jax.lax.rsqrt(ms2 + 1e-5)
    ff = jnp.maximum(h_norm @ w_ff, 0)
    return h + ff
