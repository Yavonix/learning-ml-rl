import jax_dataloader as jdl
from jax import numpy as jnp
import jax
import numpy as np
import torchvision
import wandb
from typing import Any
from PIL import Image
import optax
from flax import nnx
import orbax.checkpoint as ocp
from dataset import SingleFolderDataset, show_image

from model import Encoder_Decoder
from pathlib import Path

##  wandb login --help
## wandb: Currently logged in as: roman to http://localhost:8080.

run = wandb.init( # ctrl m on local ui for dark mode
    entity="yavonix",
    project="2802ict",
    job_type="cnn-train",
    dir="./wandb-logs",
    mode="offline"
)

path = Path('/home/roman/learning-ml/2802ict-assignments/assignment_01/maze_src/cnn_heuristic/checkpoints/')
# path = ocp.test_utils.erase_and_create_empty('/home/roman/learning-ml/2802ict-assignments/assignment_01/maze_src/cnn_heuristic/checkpoints/')

config: Any = run.config
config.seed = 0
config.batch_size = 32
config.learning_rate = 1e-2
config.epochs = 100

rngs = nnx.Rngs(params=config.seed)

## Helpers

def show_image(img: np.ndarray):
    image = Image.fromarray(img)
    image.show()

## Dataset loading

pt_ds_train = SingleFolderDataset("./motion_planning_datasets/mazes/train")
pt_ds_val = SingleFolderDataset("./motion_planning_datasets/mazes/validation")

jdl.manual_seed(0)
dl_train = jdl.DataLoader(pt_ds_train, 'pytorch', batch_size=config.batch_size, shuffle=True)
dl_val = jdl.DataLoader(pt_ds_val, 'pytorch', batch_size=config.batch_size, shuffle=False)


## Loss definition

def loss_fn(model, batch: tuple[jnp.ndarray, jnp.ndarray]):
    X, y = batch
    y_pred = model(X)
    assert(y_pred.shape == y[...,0:1].shape)
    loss: jnp.ndarray = optax.l2_loss(y_pred, y[...,0:1]) * y[...,1:2] # pyright: ignore[reportAssignmentType]
    loss = loss.sum() / y[...,1].sum()
    return loss, y_pred

@nnx.jit
def train_step(model: nnx.Module, optimiser: nnx.Optimizer, metrics: nnx.MultiMetric, batch: tuple[jax.Array, jax.Array]):
    (loss, y_pred), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    metrics.update(loss=loss)
    optimiser.update(grads)

@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch: tuple[jax.Array, jax.Array]):
  loss, y_pred = loss_fn(model, batch)
  metrics.update(loss=loss)

## Model initialisation

model = Encoder_Decoder(rngs)

# tx = optax.chain(optax.sgd(config.learning_rate))
tx = optax.chain(optax.adam(config.learning_rate))
optimiser = nnx.Optimizer(model, tx)

metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average("loss"),
    # acc = nnx.metrics.Accuracy()
)

## Train loop

for i in range(config.epochs):
    print(f"epoch {i+1} of {config.epochs}")

    model.train()
    for batch in dl_train:
        train_step(model, optimiser, metrics, batch)

    m = metrics.compute()
    # print(m)
    metrics.reset()
    run.log({"train/loss": m["loss"]})

    model.eval()
    for batch in dl_val:
        eval_step(model, metrics, batch)

    m = metrics.compute()
    print(m)
    metrics.reset()
    run.log({"val/loss": m["loss"]})

# lets log some images and our classifications:

checkpointer = ocp.StandardCheckpointer()
graph, state = nnx.split(model)



checkpointer.save(path / 'save-no-normalisation-100/', state)

checkpointer.wait_until_finished()

# X, y = next(iter(dl_train))

# y_pred = model(X)

# show_image(X[0])
# show_image(y_pred[0])

# X = np.squeeze(X, -1)

# y_pred = nnx.softmax(y_pred, axis=-1)
# y_pred = y_pred.argmax(axis=-1)
# images = [Image.fromarray(image*255).convert("L") for image in X]
# labels = y_pred

# wandb.log({"examples": [wandb.Image(image, caption=label) for (image, label) in zip(images, labels)]})

run.finish()
