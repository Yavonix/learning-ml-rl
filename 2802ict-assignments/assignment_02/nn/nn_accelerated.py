import numpy as np
import sys
from typing import Iterator
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp
from jax import random
from functools import partial

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y_onehot = np.zeros((y.shape[0], n_classes))
    for i, label in enumerate(y):
        y_onehot[i, int(label)] = 1
    return y_onehot

# I'm just more used to working with training data in this format
# normalise and one hot encode
def sep_labels(x: np.ndarray, n_classes: int) -> tuple[jax.Array, jax.Array]:
    labels = one_hot(x[:, 0], n_classes)
    features = x[:, 1:] / 256
    return (jnp.array(features), jnp.array(labels))

def batch(data: tuple[jax.Array, jax.Array], batch_size: int) -> Iterator[tuple[jax.Array, jax.Array]]:
    for start in range(0, data[0].shape[0], batch_size):
        end = min(data[0].shape[0], start + batch_size)
        yield (data[0][start:end], data[1][start:end])


def to_device_batches(x_np, y_np, batch_size): # will drop any remainder
    x = jnp.array(x_np, dtype=jnp.float32)
    y = jnp.array(y_np, dtype=jnp.float32)
    n = (x.shape[0] // batch_size) * batch_size
    x = x[:n]
    y = y[:n]
    xb = x.reshape(n // batch_size, batch_size, -1)   # (B, N, D)
    yb = y.reshape(n // batch_size, batch_size, -1)   # (B, N, C)
    return xb, yb

@jax.tree_util.register_dataclass
@dataclass
class Model:
    W_1: jax.Array
    B_1: jax.Array
    W_2: jax.Array
    B_2: jax.Array

@jax.tree_util.register_dataclass
@dataclass
class RunningAverage:
    value: jax.Array = field(default_factory=lambda: jnp.array(0.0, jnp.float32))
    count: jax.Array = field(default_factory=lambda: jnp.array(0, jnp.int32))

    def update(self, new_value):
        n_value = (self.value * self.count + new_value)/(self.count+1)
        n_count = self.count + 1
        return RunningAverage(n_value, n_count)
    def reset(self):
        return RunningAverage(jnp.array(0.0, jnp.float32), jnp.array(0, jnp.int32))

@jax.tree_util.register_dataclass
@dataclass
class ModelMetrics:
    loss: RunningAverage = field(default_factory=RunningAverage)
    accuracy: RunningAverage = field(default_factory=RunningAverage)

    def update(self, y: jax.Array, y_hat: jax.Array):
        # loss
        L = y * y_hat
        L = L.sum(axis=1)
        L = -jnp.log(L + 1e-8)
        L = L.mean()
        loss = self.loss.update(L)

        # accuracy
        pred = y_hat.argmax(axis=1)
        true = y.argmax(axis=1)
        accuracy = (pred == true).mean()
        accuracy = self.accuracy.update(accuracy)

        return ModelMetrics(loss, accuracy)

    def reset(self):
        return ModelMetrics(RunningAverage(), RunningAverage())

def init_model(key, n_input, n_hidden, n_output) -> Model:
    key1, key2, key3, key4 = random.split(key, 4)
    W_1 = jax.random.normal(key1, (n_hidden, n_input)) * 0.01
    B_1 = jax.random.normal(key2, (n_hidden)) * 0.01
    W_2 = jax.random.normal(key3, (n_output, n_hidden)) * 0.01
    B_2 = jax.random.normal(key4, (n_output)) * 0.01
    return Model(W_1, B_1, W_2, B_2)

@jax.jit
def train_step(batch: tuple[jax.Array, jax.Array], m: Model, mm: ModelMetrics, lr: float):
    print("compiling train_step")
    # the logic behind this function can be found in planning.ipynb/planning.pdf  
    n_examples = batch[0].shape[0]
    X = batch[0]
    y = batch[1]

    # forward pass
    Z_1 = X@m.W_1.T+m.B_1
    H_1 = sigmoid(Z_1)
    Z_2 = H_1@m.W_2.T+m.B_2

    # softmax
    adj_Z_2 = jnp.exp(Z_2 - Z_2.max(axis=1)[:, jnp.newaxis])
    adj_Z_2_sum = jnp.sum(adj_Z_2, 1)[:, jnp.newaxis]
    y_hat = adj_Z_2 / adj_Z_2_sum

    # backprop layer 2
    Delta_2 = y_hat - y
    grad_W2 = (Delta_2.T @ H_1) / n_examples
    grad_B2 = Delta_2.mean(axis=0)
    
    # backprop layer 1
    S_1 = sigmoid(Z_1)
    Delta_1 = (Delta_2 @ m.W_2) * S_1 * (1 - S_1)
    grad_W1 = (Delta_1.T @ X) / n_examples
    grad_B1 = Delta_1.mean(axis=0)

    # update weights
    W_2 = m.W_2 - lr * grad_W2
    B_2 = m.B_2 - lr * grad_B2
    W_1 = m.W_1 - lr * grad_W1
    B_1 = m.B_1 - lr * grad_B1
    m = Model(W_1, B_1, W_2, B_2)

    mm = mm.update(y, y_hat)

    return (m, mm)

@jax.jit
def eval(batch: tuple[jax.Array, jax.Array], m: Model, mm: ModelMetrics):
    print("compiling eval")
    X = batch[0]
    y = batch[1]

    # forward pass
    Z_1 = X@m.W_1.T+m.B_1
    H_1 = sigmoid(Z_1)
    Z_2 = H_1@m.W_2.T+m.B_2

    # softmax
    adj_Z_2 = jnp.exp(Z_2 - Z_2.max(axis=1)[:, jnp.newaxis])
    adj_Z_2_sum = jnp.sum(adj_Z_2, 1)[:, jnp.newaxis]
    y_hat = adj_Z_2 / adj_Z_2_sum

    mm = mm.update(y, y_hat)

    return mm

@jax.jit
def run_all_evals(m, mm, xb, yb):
    print("compiling run_evals")
    def body(carry, t):
        mm = carry
        x = xb[t]
        y = yb[t]
        mm = eval((x, y), m, mm)
        return mm, None
    mm, _ = jax.lax.scan(body, mm, jnp.arange(xb.shape[0]))
    return mm

@jax.jit
def run_epoch(m, mm, xb, yb, lr):
    print("compiling run_epoch")
    def body(carry, t):
        m, mm = carry
        x = xb[t]
        y = yb[t]
        m, mm = train_step((x, y), m, mm, lr)
        return (m, mm), None
    (m, mm), _ = jax.lax.scan(body, (m, mm), jnp.arange(xb.shape[0]))
    return m, mm

@partial(jax.jit, static_argnames=("epochs"))
def run_all_epochs(m, mm, xb, yb, epochs, lr):
    print("compiling run_all_epochs")
    def body(carry, t):
        m, mm = carry
        m, mm = run_epoch(m, mm, xb, yb, lr)
        return (m, mm), None
    (m, mm), _ = jax.lax.scan(body, (m, mm), jnp.arange(epochs))
    return m, mm

if __name__ == "__main__":

    n_input = int(sys.argv[1])
    n_hidden = int(sys.argv[2])
    n_output = int(sys.argv[3])

    classes = 10
    lr = 0.02
    epochs = 100

    d_train = sep_labels(np.loadtxt(sys.argv[4], delimiter=',', skiprows=1), classes) # (60000, 785) first col is label
    d_test = sep_labels(np.loadtxt(sys.argv[5], delimiter=',', skiprows=1), classes) # (10000, 785) first col is label

    key = random.key(0)
    model = init_model(key, n_input, n_hidden, n_output)

    metrics = ModelMetrics()

    batched_d_train = to_device_batches(d_train[0], d_train[1], 32)
    print(batched_d_train[0].shape, batched_d_train[1].shape)

    # model, metrics = run_all_epochs(model, metrics, batched_d_train[0], batched_d_train[1], epochs, lr)
    # print(f"l={metrics.loss.value} acc={metrics.accuracy.value}")
    # metrics = metrics.reset()


    for i in range(epochs):
        loss = 0
        # mb mini batch
        model, metrics = run_epoch(model, metrics, batched_d_train[0], batched_d_train[1], lr)
        print(f"l={metrics.loss.value} acc={metrics.accuracy.value}")
        metrics = metrics.reset()