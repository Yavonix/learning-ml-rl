from jax import numpy as jnp
import jax
import matplotlib.pyplot as plt
import time



x = jnp.linspace(0.0, 20.0, 100_000)

def f(x):
    return jnp.sin(x)

# 3. Make the grad function and (optionally) jit it
grad_f = jax.jit(jax.grad(lambda x: f(x).sum()))

# 4. Compute all the cosines in one go
cos_vals = grad_f(x)

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
plt.plot(x, f(x), x, cos_vals)  # Plot some data on the Axes.
plt.show()  


# ## VMAP Approach

# # We create a linear space array
# x: jax.Array = jnp.linspace(0.0, 20.0, 100_000)

# # We define our function
# y = lambda x: jnp.sin(x)
# x_grad_fn = jax.grad(y)

# # We compile our function

# fused_grad = jax.jit(jax.vmap(jax.grad(y)))
# fused_grad(x).block_until_ready()

# # We compute the derivative of our function


# start = time.time()
# x_grads = fused_grad(x)
# end = time.time()
# print("JIT:", end - start)

# fig, ax = plt.subplots()             # Create a figure containing a single Axes.
# plt.plot(x, y(x), x, x_grads)  # Plot some data on the Axes.
# plt.show()                