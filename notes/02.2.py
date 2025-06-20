from d2l import jax as d2l

from jax import numpy as jnp


# X = jnp.arange(2*3*4).reshape((2, 3, 4))  # Create a 2x3x4 array

# print(X, X.shape)

# print(X.sum(axis=0).shape, X.sum(axis=0))
# print(X.sum(axis=1).shape, X.sum(axis=1))
# print(X.sum(axis=2).shape, X.sum(axis=2))

# Element wise product is called Hadamard product
# Y = jnp.arange(2*3*4).reshape((4, 3, 2))

# print(X * Y)  # Element-wise multiplication (Hadamard product)

# A = jnp.arange(100*200).reshape((100, 200))
# B = jnp.arange(100*200).reshape((100, 200))
# C = jnp.arange(100*200).reshape((100, 200))

# K = jnp.stack((A, B, C), axis=0)
# K2 = jnp.array([A, B, C])

# print(K.shape, K2.shape)  # Should print (3, 100, 200)

