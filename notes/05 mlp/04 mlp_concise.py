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

##  wandb login --help
## wandb: Currently logged in as: roman to http://localhost:8080.

run = wandb.init( # ctrl m on local ui for dark mode
    entity="roman",
    project="learning-jax-mnist",
    job_type="simple-demo",
    dir="./wandb-logs",
    mode="disabled"
)

config: Any = run.config
config.seed = 0
config.batch_size = 256
config.learning_rate = 1e-1
config.epochs = 10

rngs = nnx.Rngs(params=config.seed)

## Helpers

def show_image(img: np.ndarray):
    image = Image.fromarray(img)
    image.show()

def text_labels(indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[i] for i in indices] 

# incredibly unstable
def softmax(X: jnp.ndarray):
    X_exp = jnp.exp(X)
    partition = X_exp.sum(axis=-1, keepdims=True)
    ret = X_exp / partition
    return ret

## Dataset loading

pt_ds_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)), lambda img: np.array(img, dtype=float) / 255.0])
pt_ds_train = torchvision.datasets.MNIST("/var/tmp/mnist/", download=True, transform=pt_ds_transform, train=True)
pt_ds_val = torchvision.datasets.MNIST("/var/tmp/mnist/", download=True, transform=pt_ds_transform, train=False)

jdl.manual_seed(0)
dl_train = jdl.DataLoader(pt_ds_train, 'pytorch', batch_size=config.batch_size, shuffle=True)
dl_val = jdl.DataLoader(pt_ds_val, 'pytorch', batch_size=config.batch_size, shuffle=False)

## Model definition

class MLPClassifier(nnx.Module):
    def __init__(self, din, hidden, dout, rngs: nnx.Rngs):
        self.din = din
        self.dout = dout
        self.linear1 = nnx.Linear(din, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, dout, rngs=rngs)

    def __call__(self, X: jax.Array):
        X = X.reshape((X.shape[0], -1))
        H = self.linear1(X)
        return self.linear2(nnx.tanh(H))

## Loss definition

def loss_fn(model, batch):
    X, y = batch
    y_pred = model(X)
    loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()
    return loss, y_pred

@nnx.jit
def train_step(model: nnx.Module, optimiser: nnx.Optimizer, metrics: nnx.MultiMetric, batch: tuple[jax.Array, jax.Array]):
    (loss, y_pred), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    metrics.update(loss=loss, logits=y_pred, labels=batch[1])
    optimiser.update(grads)

@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch: tuple[jax.Array, jax.Array]):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch[1])

## Model initialisation

res = []

# for k in range(4,12):

model = MLPClassifier(784, 1024, 10, rngs)

tx = optax.chain(optax.sgd(config.learning_rate))
optimiser = nnx.Optimizer(model, tx)

metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average("loss"),
    acc = nnx.metrics.Accuracy()
)

## Train loop

# print(f"Fitting with {2**k} hidden layers")
# m = {}

for i in range(config.epochs):
    print(f"epoch {i+1} of {config.epochs}")

    for batch in dl_train:
        train_step(model, optimiser, metrics, batch)

    m = metrics.compute()
    metrics.reset()
    # run.log({"train/loss": m["loss"], "train/acc": m["acc"]})

    for batch in dl_val:
        eval_step(model, metrics, batch)

    m = metrics.compute()
    print(m)
    metrics.reset()
    # run.log({"val/loss": m["loss"], "val/acc": m["acc"]})
    
    # m['hidden'] = 2**k

    # res.append(m)

# for entry in res:
#     print(entry)
    

# lets log some images and our classifications:

# X, y = next(iter(dl_train))

# y_pred = model(X)
# y_pred = nnx.softmax(y_pred, axis=-1)
# y_pred = y_pred.argmax(axis=-1)
# images = [Image.fromarray(image*255).convert("L")for image in X]
# labels = text_labels(y_pred)

# wandb.log({"examples": [wandb.Image(image, caption=label) for (image, label) in zip(images, labels)]})

# run.finish()
