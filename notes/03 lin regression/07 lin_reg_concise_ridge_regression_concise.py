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
config.weight_decay = 5

rngs = nnx.Rngs(params=config.seed)

def syntheticRegressionData(w: jax.Array, b: float, n: int, rngs: nnx.Rngs):
    X = jax.random.normal(rngs.params(), (n, w.shape[0]))
    noise = jax.random.normal(rngs.params(), (n, 1)) * 0.01
    y = (X * w).sum(axis=1, keepdims=True) + b + noise
    return (X, y)

X, y = syntheticRegressionData(w=jnp.array([2, -3.4]), b=4.2, n=100000, rngs=rngs)

arr_ds = jdl.ArrayDataset(X, y)
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=config.batch_size, shuffle=True)

def l2_penalty(w):
    return (w ** 2).sum() / 2

def loss_fn(model: nnx.Linear, batch):
    x, y = batch
    y_pred = model(x)
    assert y_pred.shape == y.shape
    return optax.l2_loss(y_pred, y).mean()
    # return optax.l2_loss(y_pred, y).mean() + optax.l2_loss()


@nnx.jit
def train_step(model: nnx.Module, optimiser: nnx.Optimizer, batch: Tuple[jax.Array, jax.Array]):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimiser.update(grads)
    return loss

model = nnx.Linear(2, 1, kernel_init=nnx.initializers.normal(0.01), rngs=rngs)

# Masking weight decay to only apply to the kernal:

# We can do this using nnx filters: https://github.com/google/flax/discussions/4737

# Or construct a pytree tree

# def make_decay_mask(tree):
#     def mask_fn(path, leaf):
#         # path is a tuple of keys, e.g. (DictKey(key='kernel'), GetAttrKey(name='value'))
#         print(path)
#         # Check if the first key is DictKey(key='kernel')
#         return hasattr(path[0], 'key') and path[0].key == 'kernel'
#     return jax.tree_util.tree_map_with_path(mask_fn, tree)
# _, params = nnx.split(model, nnx.Param)
# kernel_only = make_decay_mask(params)
# tx = optax.chain(
#     optax.add_decayed_weights(config.weight_decay, mask=kernel_only)
#     optax.sgd(config.learning_rate),
# )

# Or pass a function which transforms the pytree
# It is common to skip weight decay for BatchNorm scale and all bias parameters.
# Since in many networks, these are the only 1D parameters, you may for instance
# create a mask function to mask them out as follows:

# mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p) 

# tx = optax.chain(
#     optax.add_decayed_weights(config.weight_decay, mask_fn)
#     optax.sgd(config.learning_rate),
# )

# Another variation of the above:

mask_fn = lambda p: jax.tree.map(lambda x: x.ndim != 1, p)
decay_weights = optax.masked(optax.add_decayed_weights(config.weight_decay), mask_fn)

# so decay weights needs to go before sgd
# decay weights is a gradient transform:
# - adds a bit to each weight: g' = g+λw
# sgd is an update transform:
# - takes transformed gradients to produce the actual parameter delta Δw
# - Δw=−ηg’
# - always run LAST in the chain


tx = optax.chain(
    decay_weights,
    optax.sgd(config.learning_rate)
)

# tx = optax.chain(
#     optax.sgd(config.learning_rate),
# )

optimiser = nnx.Optimizer(model, tx)

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