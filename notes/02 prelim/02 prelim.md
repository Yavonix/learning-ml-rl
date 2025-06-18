## Data Manip

oOk.

## Linear Algebra

Thinking about summing axis geometrically seems really difficult. Im going to stick to thinking about summing in an element wise fashion.

## Calculus

The following rules come in handy 
for differentiating multivariate functions:

* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$ we have $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ and $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* For square matrices $\mathbf{A} \in \mathbb{R}^{n \times n}$ we have that $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ and in particular
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Similarly, for any matrix $\mathbf{X}$, 
we have $\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$. 

## Autograd

1. Second derivate contains mixed partial derivatives. 
2. Ok.
3. Jacobian outputs (have to call jax.jacfwd or jax.jacrev)
4. Below is was what I came up with. Alternatively we can collapse into a sum to avoid using vmap.

```py
from jax import numpy as jnp
import jax
import matplotlib.pyplot as plt
import time

# We create a linear space array
x: jax.Array = jnp.linspace(0.0, 20.0, 100_000)

# We define our function
y = lambda x: jnp.sin(x)
x_grad_fn = jax.grad(y)

# We compile our function

fused_grad = jax.jit(jax.vmap(jax.grad(y)))
fused_grad(x).block_until_ready()

# We compute the derivative of our function


start = time.time()
x_grads = fused_grad(x)
end = time.time()
print("JIT:", end - start)

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
plt.plot(x, y(x), x, x_grads)  # Plot some data on the Axes.
plt.show()                
```

5. Already done.
6. Already done.
7. Im not doing that
8. Forward mode is optimal when you have many outputs while backward mode is optimal when you have many inputs. Has to do with the order of matrix ops when computing the jacobian.