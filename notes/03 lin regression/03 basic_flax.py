from flax import nnx
import jax
import jax.numpy as jnp
# from jaxtyping import Array, Float, PyTree

class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        # Dynamic values must be a param or variable
        # variable used for stateful computation
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        return x @ self.w + self.b


class MLP(nnx.Module):
    def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
        ## nnx modules can be assigned directly
        self.linear1 = Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs) # stateful module that stores nnx.Rngs
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = Linear(dmid, dout, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)

