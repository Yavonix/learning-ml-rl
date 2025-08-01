from typing import Any
import wandb
import random

run = wandb.init(
    entity="roman",
    project="learning-jax",
    job_type="simple-demo",
    dir="./wandb-logs"
)

config: Any = run.config
config.seed = 42
config.batch_size = 64
config.validation_split = 0.2
config.pooling = "avg"
config.learning_rate = 1e-4
config.epochs = 15

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()