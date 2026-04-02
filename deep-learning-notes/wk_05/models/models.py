from flax import nnx
from functools import partial
from flax.typing import PaddingLike
from jax import eval_shape, ShapeDtypeStruct, numpy as jnp
from math import prod

class AvgPool(nnx.Module):
    def __init__(self, window_shape, strides, padding="VALID"):
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        return nnx.avg_pool(x, self.window_shape, self.strides, self.padding)

class MaxPool(nnx.Module):
    def __init__(self, window_shape, strides, padding="VALID"):
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding

    def __call__(self, x):
        return nnx.max_pool(x, self.window_shape, self.strides, self.padding)

class LeNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, image_size=28, num_classes=10):
        self.features = nnx.Sequential(
            nnx.Conv(1, 6, (5, 5), padding="SAME", rngs=rngs),
            nnx.sigmoid,
            AvgPool((2,2), (2,2)),
            nnx.Conv(6, 16, (5, 5), padding="VALID", rngs=rngs),
            nnx.sigmoid,
            AvgPool((2,2), (2,2))
        )

        dummy_data = ShapeDtypeStruct((1, image_size, image_size, 1), jnp.float32)
        dummy_shape = eval_shape(self.features, dummy_data).shape
        flattened_size = prod(dummy_shape[1:])

        self.classifier = nnx.Sequential(
            nnx.Linear(flattened_size, 120, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(120, 84, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(84, num_classes, rngs=rngs)
        )

    def __call__(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

class AlexNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, image_size=224, num_classes=100):
        self.features = nnx.Sequential(
            nnx.Conv(3, 96, (11,11), (4,4), padding=(1,1), rngs=rngs), nnx.relu,
            MaxPool((3,3), (2,2)),
            nnx.Conv(96, 256, (5,5), padding="SAME", rngs=rngs), nnx.relu,
            MaxPool((3,3), (2,2)),
            nnx.Conv(256, 384, (3,3), padding=(1,1), rngs=rngs), nnx.relu,
            nnx.Conv(384, 384, (3,3), padding=(1,1), rngs=rngs), nnx.relu,
            nnx.Conv(384, 256, (3,3), padding=(1,1), rngs=rngs), nnx.relu,
            MaxPool((3,3), (2,2)),
        )

        dummy_data = ShapeDtypeStruct((1, image_size, image_size, 3), jnp.float32)
        dummy_shape = eval_shape(self.features, dummy_data).shape
        flattened_size = prod(dummy_shape[1:])

        self.classifier = nnx.Sequential(
            nnx.Linear(flattened_size, 4096, rngs=rngs), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, 4096, rngs=rngs), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, num_classes, rngs=rngs)
        )

    def __call__(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

class VGG_Block(nnx.Module):
    def __init__(self, num_convs, in_channels, out_channels, rngs: nnx.Rngs):
        layers = []
        for i in range(num_convs):
            layers.append(nnx.Conv(in_channels if i == 0 else out_channels, out_channels, kernel_size=(3,3), padding="SAME", rngs=rngs))
            layers.append(nnx.relu)
        self.layers = nnx.Sequential(*layers, MaxPool((2,2), (2,2)))
    
    def __call__(self, x):
        return self.layers(x)

class VGG(nnx.Module):
    def __init__(self, arch: list[tuple[int,int]], rngs: nnx.Rngs, image_size, in_channels=3, num_classes=10):
        layers = []
        temp_in_channels = in_channels
        for (num_convs, out_channels) in arch:
            layers.append(VGG_Block(num_convs, temp_in_channels, out_channels, rngs=rngs))
            temp_in_channels = out_channels
        
        self.features = nnx.Sequential(*layers)
        
        dummy_data = ShapeDtypeStruct((1, image_size, image_size, in_channels), jnp.float32)
        dummy_shape = eval_shape(self.features, dummy_data).shape
        flattened_size = prod(dummy_shape[1:])

        self.classifier = nnx.Sequential(
            nnx.Linear(flattened_size, 4096, rngs=rngs), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, 4096, rngs=rngs), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, num_classes, rngs=rngs)
        )

    def __call__(self, x: jnp.ndarray):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

class NiN_Block(nnx.Module):
    def __init__(self, in_features, out_features, kernel_size, strides, rngs: nnx.Rngs, padding:PaddingLike = "VALID"):
        self.layers = nnx.Sequential(
            nnx.Conv(in_features, out_features, kernel_size=kernel_size, strides=strides, padding=padding, rngs=rngs),
            nnx.relu,
            nnx.Conv(out_features, out_features, (1,1), rngs=rngs),
            nnx.relu,
            nnx.Conv(out_features, out_features, (1,1), rngs=rngs),
            nnx.relu
        )

    def __call__(self, x):
        return self.layers(x)

class NiN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, num_classes=10):
        self.layers = nnx.Sequential(
            NiN_Block(3, 96, (11,11), (4,4), rngs=rngs),
            MaxPool((3,3), (2,2)),
            NiN_Block(96, 256, (5,5), (1,1), rngs=rngs, padding="SAME"),
            MaxPool((3,3), (2,2)),
            NiN_Block(256, 384, (3,3), (1,1), rngs=rngs, padding="SAME"),
            MaxPool((3,3), (2,2)),
            nnx.Dropout(0.5, rngs=rngs),
            NiN_Block(384, num_classes, (3,3), (1,1), rngs=rngs, padding="SAME"),
        )

    def __call__(self, x: jnp.ndarray):
        x = self.layers(x)
        x = x.mean(axis=(1,2)) # NHWC; C=10
        return x