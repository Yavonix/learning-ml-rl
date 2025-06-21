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

@nnx.jit
def train_step(model: nnx.Module, optimiser: nnx.Optimizer, batch: Tuple[jax.Array, jax.Array]):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimiser.update(grads)
    return loss


# path = ocp.test_utils.erase_and_create_empty('/home/roman/learning-ml/checkpoints')
path = '/home/roman/learning-ml/checkpoints'
options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2)
mngr = ocp.CheckpointManager(
    path, options=options
)

## Load checkpointed model
latest_step = mngr.latest_step()

model: nnx.Linear

if (latest_step == None): 
    model: nnx.Linear = nnx.Linear(2, 1, rngs=rngs)
    latest_step = 0
else:
    abstract_model: nnx.Linear = nnx.eval_shape(lambda: nnx.Linear(2, 1, rngs=nnx.Rngs(config.seed)))
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = mngr.restore(
        mngr.latest_step(),
        args=ocp.args.StandardRestore(abstract_state),
    )
    model: nnx.Linear = nnx.merge(graphdef, state_restored)

optimiser = nnx.Optimizer(model, optax.sgd(config.learning_rate))

for i in range(latest_step, latest_step + config.epochs):
    st = time.time()
    ls: list[float] = []
    for batch in dataloader:
        loss = train_step(model, optimiser, batch)
        ls.append(loss)
        run.log({"train/batch_loss": loss})
    run.log({"train/epoch_loss": sum(ls)/len(ls)})

    _, model_tree = nnx.split(model)

    mngr.save(i, args=ocp.args.StandardSave(model_tree))

    print(f"Epoch {i} in {time.time()-st:0.2f} sec")

mngr.wait_until_finished()

print(f"w={model.kernel} b={model.bias}")