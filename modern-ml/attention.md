# Terminology

- Input axes (`I`): Determine the depth of each filter.
- Output axes (`O`): Determine the number of filters per group.
- Group axes (`G`): Determine the number of independent groups of filters.

The tensor order of $W$ is $I + O + G$

# Linear layer

For `...hd,hde->...e`

We have
- 2 input axes (`hd`)
- 1 output axes (`e`)
- kernel is `hde`

Input features determine the size/depth of each filter
Output features determine the number of filters

When an axes is not shared between the input and output: `kernel.ndim = I + O + G`

```python
layer = nnx.LinearGeneral(
    in_features=(H, D),
    out_features=E,
    axis=(-2, -1),
    use_bias=False,
    rngs=rngs,
)
Y = layer(X)

layer = nnx.Einsum(
    "...hd,hde->...e",
    kernel_shape=(H, D, E),
    rngs=rngs,
)
Y = layer(X)

# Or you can just flatten
layer = nnx.Linear(H * D, E, use_bias=False, rngs=rngs)
Y = layer(X.reshape(B, H * D))
```

# Grouped linear layer

Where we want to learn a separate kernel for each index of an input

For `bgd,gde->bge`

We have
- 1 input axes (`d`)
- 1 output axes (`e`)
- 1 group axes (`g`)

We repeat `g` meaning we learn a separate kernel for each index of g


```python
layer = nnx.Einsum(
    "bgd,gde->bge",
    kernel_shape=(G, D, E),
    rngs=rngs,
)
Y = layer(X)

# equivalent LinearGeneral
layer = nnx.LinearGeneral(
    in_features=D,
    out_features=E,
    batch_axis={0: G},
    rngs=rngs,
)
# batch axis must start at 0
Y = layer(jnp.moveaxis(X, 1, 0))
Y = jnp.moveaxis(Y, 0, 1)
```

# Examples

`bd,de->be`: Standard linear projection
`bd,dhe->bhe`: Linear projection with multiple output axes
`bgd,gde->bge`: Group-specific linear projection


```python
nnx.Einsum(
    "bgd,gde->bge",
    kernel_shape=(G, D, E),
    bias_shape=(G, E),
    rngs=rngs,
)

import jax
import jax.numpy as jnp
from flax import nnx

# ... = batch dimensions
# q = query sequence length
# k = key/value sequence length
# h = number of attention heads
# d = features per head
# e = embedding dimension (d_model)

def scaled_dot_product_attention(Q, K, V, *, is_causal=False):
    # Q: (..., Q, H, D)
    # K, V: (..., K, H, D)

    scores = jnp.einsum("...qhd,...khd->...hqk", Q, K)
    scores /= Q.shape[-1] ** 0.5

    if is_causal:
        Q_len, K_len = scores.shape[-2:]
        if Q_len != K_len:
            raise ValueError("Causal masking requires Q == K here")

        mask = jnp.tril(jnp.ones((Q_len, K_len), dtype=bool))
        scores = jnp.where(mask, scores, -jnp.inf)

    weights = jax.nn.softmax(scores, axis=-1)

    return jnp.einsum("...hqk,...khd->...qhd", weights, V)


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model, n_heads, *, rngs):
        assert d_model % n_heads == 0
        d_head = d_model // n_heads

        projection = lambda : nnx.Einsum("...e,ehd->...hd",
            kernel_shape=(d_model, n_heads, d_head),
            bias_shape=(n_heads, d_head),
            rngs=rngs,
        )

        self.W_q = projection()
        self.W_k = projection()
        self.W_v = projection()

        self.W_o = nnx.Einsum(
            "...hd,hde->...e",
            kernel_shape=(n_heads, d_head, d_model),
            bias_shape=(d_model,),
            rngs=rngs,
        )

    def __call__(self, x, context=None, *, is_causal=False):
        if context is None:
            context = x

        Q = self.W_q(x)
        K = self.W_k(context)
        V = self.W_v(context)

        y = scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal
        )

        return self.W_o(y)


attn = MultiHeadAttention(64, 4, rngs=nnx.Rngs(0))

x = jnp.ones((2, 10, 64))
memory = jnp.ones((2, 20, 64))

# Bidirectional self-attention
y = attn(x)                          # (2, 10, 64)

# Causal self-attention (e.g. GPT)
y = attn(x, is_causal=True)          # (2, 10, 64)

# Cross-attention (e.g. encoder-decoder)
y = attn(x, context=memory)          # (2, 10, 64)

```