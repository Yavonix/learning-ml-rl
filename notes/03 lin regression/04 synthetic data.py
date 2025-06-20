import time
from typing import Tuple
from jax import numpy as jnp
import jax
import jax_dataloader as jdl

from flax import nnx

rng = nnx.Rngs(params=0)

def syntheticRegressionData(w: jax.Array, b: float, n: int, rng: nnx.Rngs):
    X = jax.random.normal(rng.params(), (n, w.shape[0]))
    noise = jax.random.normal(rng.params(), (n, 1)) * 0.01
    y = (X * w).sum(axis=1, keepdims=True) + b + noise
    return (X, y)

X, y = syntheticRegressionData(w=jnp.array([2, -3.4]), b=4.2, n=100000, rng=rng)

arr_ds = jdl.ArrayDataset(X, y)
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=32, shuffle=True)

for epoch in range(100):
  start_time = time.time()
  for x, y in dataloader:
    ## do some math
    pass
  epoch_time = time.time() - start_time

  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))