import jax_dataloader as jdl
from jax import numpy as jnp
import numpy as np
import torchvision
import wandb
from typing import Any
from PIL import Image

run = wandb.init( # ctrl m on local ui for dark mode
    entity="roman",
    project="learning-jax-mnist",
    job_type="simple-demo",
    dir="./wandb-logs",
    mode="disabled"
)

config: Any = run.config
config.seed = 0
config.batch_size = 32
config.learning_rate = 1e-3
config.epochs = 10

## Helpers

def show_image(img: np.ndarray):
    image = Image.fromarray(img)
    image.show()

def text_labels(indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[i] for i in indices]

## Dataset loading

pt_ds_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28,28)), lambda img: np.array(img, dtype=float)])
pt_ds_train = torchvision.datasets.FashionMNIST("/tmp/mnist/", download=True, transform=pt_ds_transform, train=True)
pt_ds_val = torchvision.datasets.FashionMNIST("/tmp/mnist/", download=True, transform=pt_ds_transform, train=False)

jdl.manual_seed(0)
dl_train = jdl.DataLoader(pt_ds_train, 'pytorch', batch_size=config.batch_size, shuffle=True)
dl_val = jdl.DataLoader(pt_ds_val, 'pytorch', batch_size=config.batch_size, shuffle=False)

## 

for i in range(config.epochs):
    for batch in dl_val:
        X, y = batch
        image_array = X[0]
        image = Image.fromarray(image_array)
        print(f"showing {text_labels(y)[0]}")
        image.show(title=text_labels(y)[0])
        input("")
    print(i)