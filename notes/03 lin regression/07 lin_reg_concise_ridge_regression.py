from functools import partial
from typing import Any, Tuple
import time
import jax.numpy as jnp
import jax
import jax_dataloader as jdl
from flax import nnx as nnx
import optax
import wandb
from flax.training import train_state
import orbax.checkpoint as ocp

run = wandb.init( # ctrl m on local ui for dark mode
    entity="roman",
    project="learning-jax",
    job_type="simple-demo",
    dir="./wandb-logs"
)

config: Any = run.config
config.seed = 0
config.batch_size = 256
config.learning_rate = 1e-3
config.epochs = 10
config.weight_decay = 1e-5
 
rngs = nnx.Rngs(params=config.seed)

def syntheticRegressionData(w: jax.Array, b: float, n: int, rngs: nnx.Rngs):
    X = jax.random.normal(rngs.params(), (n, w.shape[0]))
    noise = jax.random.normal(rngs.params(), (n, 1)) * 0.01
    y = (X * w).sum(axis=1, keepdims=True) + b + noise
    return (X, y)

X, y = syntheticRegressionData(w=jnp.array([2, -3.4]), b=4.2, n=100000, rngs=rngs)

arr_ds = jdl.ArrayDataset(X, y)
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=config.batch_size, shuffle=True)

def l2_penalty(w, weight_decay):
    return weight_decay * (w ** 2).sum() / 2

def loss_fn(model: nnx.Linear, batch, weight_decay=1e-5):
    x, y = batch
    y_pred = model(x)
    assert y_pred.shape == y.shape
    return optax.l2_loss(y_pred, y).mean() + l2_penalty(model.kernel, weight_decay)
    # return optax.l2_loss(y_pred, y).mean() + optax.l2_loss()

loss_fn = partial(loss_fn, weight_decay=config.weight_decay)

@nnx.jit
def train_step(model: nnx.Module, optimiser: nnx.Optimizer, batch: Tuple[jax.Array, jax.Array]):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimiser.update(grads)
    return loss

model = nnx.Linear(2, 1, kernel_init=nnx.initializers.normal(0.01), rngs=rngs)
optimiser = nnx.Optimizer(model, optax.sgd(config.learning_rate))

start_time = time.time()

for i in range(0, config.epochs):
    st = time.time()
    ls: list[float] = []
    for batch in dataloader:
        loss = train_step(model, optimiser, batch)
        ls.append(loss)
        run.log({"train/batch_loss": loss})
    run.log({"train/epoch_loss": sum(ls)/len(ls), "epoch": i})

    _, model_tree = nnx.split(model)

    print(f"Epoch {i} in {time.time()-st:0.2f} sec")

print(f"w={model.kernel} b={model.bias}")

print(f"total time = {time.time()-start_time}")