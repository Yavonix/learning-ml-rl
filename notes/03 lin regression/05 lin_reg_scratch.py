from typing import Any, Tuple
import time
import jax.numpy as jnp
import jax
import jax_dataloader as jdl
from flax import nnx as nnx
import optax
import wandb

# Am I confused?
# Yes
# Does this work?
# Probably
# How do I know?
# GPU go brrr

run = wandb.init( # ctrl m on local ui for dark mode
    entity="roman",
    project="learning-jax",
    job_type="simple-demo",
    dir="./wandb-logs"
)

config: Any = run.config
config.seed = 0
config.batch_size = 32
# config.validation_split = 0.2
# config.pooling = "avg"
config.learning_rate = 1e-3
config.epochs = 10

rng = nnx.Rngs(params=config.seed)

def syntheticRegressionData(w: jax.Array, b: float, n: int, rng: nnx.Rngs):
    X = jax.random.normal(rng.params(), (n, w.shape[0]))
    noise = jax.random.normal(rng.params(), (n, 1)) * 0.01
    y = (X * w).sum(axis=1, keepdims=True) + b + noise
    return (X, y)

X, y = syntheticRegressionData(w=jnp.array([2, -3.4]), b=4.2, n=100000, rng=rng)

arr_ds = jdl.ArrayDataset(X, y)
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=config.batch_size, shuffle=True)

## Below for 05

class Linear(nnx.Module):
  def __init__(self, din, dout, rng):
    key = rng.params()
    # self.w = nnx.Param(jnp.zeros((din, dout)))
    self.w = nnx.Param(jax.random.normal(key, (din, dout)) * 1000) 
    self.b = nnx.Param(jnp.zeros((dout,)))
    return None

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

# class SGD:
#   def __init__(self, learning_rate: float):
#     self.learning_rate = learning_rate

#   def update(self, params, grads):
#     # Simple SGD update: param = param - lr * grad
#     return jax.tree.map(
#       lambda p, g: p - self.learning_rate * g, params, grads
#     )

# class Optimizer(nnx.Optimizer):
#   def __init__(self, model, optimizer):
#     self.model: nnx.Module = model
#     self.optimizer = optimizer

#   def update(self, grads, **kwargs):
#     # Update model parameters in-place
#     params = nnx.state(self.model, nnx.Param)
#     new_params = self.optimizer.update(params, grads)
#     nnx.update(self.model, new_params)

model = Linear(2, 1, rng)
# state = Optimizer(model, SGD(config.learning_rate))

state = nnx.Optimizer(model, optax.sgd(config.learning_rate))

def loss_fn(model, batch):
    y_pred: jax.Array = model(batch[0])
    # # print(batch[1].shape, y_pred.shape)
    # # print(y_pred, batch[1])
    # # time.sleep(5)
    # assert(batch[1].shape == y_pred.shape)
    l = (y_pred - batch[1])**2
    return l.mean()

@nnx.jit
def train_step(model: nnx.Module, state, batch: Tuple[jax.Array, jax.Array]) -> float:
  loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
  state.update(grads)
  return loss

print(model.w, model.b)

for epoch in range(config.epochs):
  start_time = time.time()
  batch_losses = []
  for batch in dataloader:
    loss = train_step(model, state, batch)
    batch_losses.append(loss)
    wandb.log({"train/batch_loss": loss})
    # epoch-level log
  
  epoch_loss = sum(batch_losses) / len(batch_losses)  
  
  wandb.log({
      "train/epoch_loss": epoch_loss,
      "epoch": epoch,
  })

  epoch_time = time.time() - start_time
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

print(model.w, model.b)