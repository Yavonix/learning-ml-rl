from flax import nnx
import jax

class Encoder(nnx.Module):
    '''
    Expects (in_features, H, W)
    Outputs (out_features, ceil(H / 2), ceil(W / 2))
    '''
    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, kernel_size=(3,3), strides=2, kernel_dilation=1, use_bias=False, padding='SAME', rngs=rngs) # downsample to 112x112
        self.norm1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=(3,3), strides=1, kernel_dilation=2, use_bias=False, padding='SAME', rngs=rngs)
        self.norm2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv3 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=(3,3), strides=1, kernel_dilation=3, use_bias=False, padding='SAME', rngs=rngs)
        self.norm3 = nnx.BatchNorm(num_features=out_features, rngs=rngs)

    def __call__(self, X: jax.Array):
        X = self.norm1(self.conv1(X))
        X = nnx.leaky_relu(X)
        X = self.norm2(self.conv2(X))
        X = nnx.leaky_relu(X)
        X = self.norm3(self.conv3(X))
        X = nnx.leaky_relu(X)
        return X


class Decoder(nnx.Module):
    '''
    Expects (in_features, H, W) # NHWC
    Outputs (out_features, ceil(H * 2), ceil(W * 2))
    '''
    def __init__(self, rngs: nnx.Rngs, in_features: int, out_features: int):
        self.deconv = nnx.ConvTranspose(in_features=in_features, out_features=out_features, kernel_size=(4,4), strides=2, use_bias=False, padding='SAME', rngs=rngs)
        self.norm1 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv1 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=(3,3), strides=1, use_bias=False, padding='SAME', rngs=rngs)
        self.norm2 = nnx.BatchNorm(num_features=out_features, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, kernel_size=(3,3), strides=1, use_bias=False, padding='SAME', rngs=rngs)
        self.norm3 = nnx.BatchNorm(num_features=out_features, rngs=rngs)

    def __call__(self, X: jax.Array):
        X = self.norm1(self.deconv(X))
        X = nnx.leaky_relu(X)
        X = self.norm2(self.conv1(X))
        X = nnx.leaky_relu(X)
        X = self.norm3(self.conv2(X))
        X = nnx.leaky_relu(X)
        return X
    
class Encoder_Decoder(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.enc1 = Encoder(rngs=rngs, in_features=3, out_features=16)
        self.enc2 = Encoder(rngs=rngs, in_features=16, out_features=32)
        self.enc3 = Encoder(rngs=rngs, in_features=32, out_features=64)

        self.dec1 = Decoder(rngs=rngs, in_features=64, out_features=32)
        self.dec2 = Decoder(rngs=rngs, in_features=32, out_features=16)
        self.dec3 = Decoder(rngs=rngs, in_features=16, out_features=16)

        self.conv1 = nnx.Conv(in_features=16, kernel_size=(1,1), out_features=1, strides=1, padding="SAME", rngs=rngs)

    def __call__(self, X: jax.Array):
        X = self.enc1(X)
        X = self.enc2(X)
        X = self.enc3(X)

        X = self.dec1(X)
        X = self.dec2(X)
        X = self.dec3(X)

        return self.conv1(X)
