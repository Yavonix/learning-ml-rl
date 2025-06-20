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

model = nnx.Linear(2, 1, rngs=rngs)
state = nnx.Optimizer(model, optax.sgd(config.learning_rate))

def loss_fn(model, batch):
    x, y = batch
    y_pred = model(x)
    return optax.l2_loss(y_pred, y).mean()

@nnx.jit
def train_step(model: nnx.Module, state: nnx.Optimizer, batch: Tuple[jax.Array, jax.Array]):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    state.update(grads)
    return loss

class Hello():
    def print(self):
        print("cat")

h = Hello()

h.print()

path = ocp.test_utils.erase_and_create_empty('/home/roman/learning-ml')
options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=2)
mngr = ocp.CheckpointManager(
    path, options=options
)

for i in range(config.epochs):
    st = time.time()
    ls: list[float] = []
    for batch in dataloader:
        loss = train_step(model, state, batch)
        ls.append(loss)
        run.log({"train/batch_loss": loss})
    run.log({"train/epoch_loss": sum(ls)/len(ls)})

    _, model_tree = nnx.split(model)

    mngr.save(i, args=ocp.args.StandardSave(model_tree))
    
    print(f"Epoch {i} in {time.time()-st:0.2f} sec")

mngr.wait_until_finished()

print(f"w={model.kernel} b={model.bias}")