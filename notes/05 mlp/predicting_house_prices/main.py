import polars as pl
import polars.selectors as cs
from typing import Any, Tuple
import time
import jax.numpy as jnp
import jax
import jax_dataloader as jdl
from flax import nnx as nnx
import optax
import wandb
from flax.training import train_state
import dataset
from sklearn.model_selection import KFold

test_df = dataset.get_test_df()

kf = KFold(n_splits=5)
arr = test_df.to_jax()

for i, (train_index, test_index) in enumerate(kf.split(arr)): # type: ignore
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    arr_ds = jdl.ArrayDataset(arr[:,0:-1], arr[:,-1])
    dl_train = jdl.DataLoader(arr_ds, 'jax', batch_size=32, shuffle=True)
print(arr.shape)