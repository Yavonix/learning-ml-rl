import jax
from jax import numpy as jnp

print(jax.default_backend())

## Creating arrays

x = jnp.arange(12) # 0 to 11

print(x, x.shape)
X = x.reshape(-1, 4)  # Reshape to 3 rows and 4 columns
                     # -1 allows automatic calculation of the size
print(X, X.shape)

jnp.zeros((3, 4))  # Create a 3x4 matrix of zeros
jnp.ones((3, 4))   # Create a 3x4 matrix of ones
jnp.eye(3)        # Create a 3x3 identity matrix
jnp.full((3, 4), 7)  # Create a 3x4 matrix filled with 7
jnp.arange(12).reshape(3, 4)  # Create a 3x4 matrix with values from 0 to 11
jnp.linspace(0, 1, 12).reshape(3, 4)  # Create a 3x4 matrix with values from 0 to 1

jax.random.normal(jax.random.PRNGKey(0), shape=(3, 4))  # Create a 3x4 matrix with random values from a normal distribution
jax.random.uniform(jax.random.PRNGKey(0), shape=(3, 4))  # Create a 3x4 matrix with random values from a uniform distribution

# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
print(X_new_1)

# Update multiple elements
X_new_2 = X.at[1:3, 3].set(17)
print(X_new_2)

jnp.exp2(jnp.array([1, 2, 3])) ## Unary scalar ops

k = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
print(k + y, k - y, k * y, k / y, k ** y)  # Binary scalar ops, element-wise operations

X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)) # We can concatenate arrays along a specified axis


# Broadcasting

## Broadcasting
#### (i) expand one or both arrays by copying elements along axes with length 1
#### so that after this transformation, the two tensors have the same shape;
#### (ii) perform an elementwise operation on the resulting arrays.

a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
print(a, b)

# Conversion

A = jax.device_get(X) # A is a standard NumPy array
B = jax.device_put(A) # B is a JAX array

a = jnp.array(3.5)
print(a, a.item(), float(a), int(a)) # we can convert a size 1 tensor to a Python scalar with .item() or by casting it to a Python type like float or int
