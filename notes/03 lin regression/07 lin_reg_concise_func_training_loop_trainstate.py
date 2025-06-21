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

rngs = nnx.Rngs(params=config.seed)

def syntheticRegressionData(w: jax.Array, b: float, n: int, rngs: nnx.Rngs):
    X = jax.random.normal(rngs.params(), (n, w.shape[0]))
    noise = jax.random.normal(rngs.params(), (n, 1)) * 0.01
    y = (X * w).sum(axis=1, keepdims=True) + b + noise
    return (X, y)

X, y = syntheticRegressionData(w=jnp.array([2, -3.4]), b=4.2, n=100000, rngs=rngs)

arr_ds = jdl.ArrayDataset(X, y)
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=config.batch_size, shuffle=True)

def loss_fn(model, batch):
    x, y = batch
    y_pred = model(x)
    return optax.l2_loss(y_pred, y).mean()

class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef

@nnx.jit
def train_step(state: TrainState, batch: Tuple[jax.Array, jax.Array]):
    model = nnx.merge(state.graphdef, state.params)
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    state = state.apply_gradients(grads=grads)
    return loss, state

model = nnx.Linear(2, 1, rngs=rngs)
start_time = time.time()

# we only traverse once at the start.
graphdef, params = nnx.split(model, nnx.Param)

state = TrainState.create(apply_fn=None,
                    graphdef=graphdef,
                    params=params,
                    tx=optax.sgd(config.learning_rate))

for i in range(0, config.epochs):
    st = time.time()
    ls: list[float] = []
    for batch in dataloader:
        loss, state = train_step(state, batch)
        ls.append(loss)
        run.log({"train/batch_loss": loss})
    run.log({"train/epoch_loss": sum(ls)/len(ls), "epoch": i})

    print(f"Epoch {i} in {time.time()-st:0.2f} sec")

nnx.update(model, state.params)

print(f"w={model.kernel} b={model.bias}")

print(f"Finished in {time.time()-start_time} sec")