import numpy as np
import sys
from typing import Iterator
from dataclasses import dataclass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y_onehot = np.zeros((y.shape[0], n_classes))
    for i, label in enumerate(y):
        y_onehot[i, int(label)] = 1
    return y_onehot

# I'm just more used to working with training data in this format
# normalise and one hot encode
def sep_labels(x: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    labels = one_hot(x[:, 0], n_classes)
    features = x[:, 1:] / 256
    print(labels.shape, features.shape)
    return (features, labels)

def batch(data: tuple[np.ndarray, np.ndarray], batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, data[0].shape[0], batch_size):
        end = min(data[0].shape[0], start + batch_size)
        yield (data[0][start:end], data[1][start:end])

@dataclass
class Model:
    W_1: np.ndarray
    B_1: np.ndarray
    W_2: np.ndarray
    B_2: np.ndarray

def init_model(n_input, n_hidden, n_output) -> Model:
    W_1 = np.random.normal(0, 0.01, (n_hidden, n_input))
    B_1 = np.random.normal(0, 0.01, n_hidden)
    W_2 = np.random.normal(0, 0.01, (n_output, n_hidden))
    B_2 = np.random.normal(0, 0.01, n_output)
    return Model(W_1, B_1, W_2, B_2)

def train(batch: tuple[np.ndarray, np.ndarray], m: Model, lr: float) -> tuple[Model, int]:
    # the logic behind this function can be found in planning.ipynb/planning.pdf
    
    n_examples = batch[0].shape[0]
    X = batch[0]
    y = batch[1]

    # forward pass
    Z_1 = X@m.W_1.T+m.B_1
    H_1 = sigmoid(Z_1)
    Z_2 = H_1@m.W_2.T+m.B_2

    # softmax
    adj_Z_2 = np.exp(Z_2 - Z_2.max(axis=1)[:, np.newaxis])
    adj_Z_2_sum = np.sum(adj_Z_2, 1)[:, np.newaxis]
    y_hat = adj_Z_2 / adj_Z_2_sum

    # loss
    L = y * y_hat
    L = L.sum(axis=1)
    L = -np.log(L)
    L = L.mean()

    # backprop layer 2
    Delta_2 = y_hat - y
    grad_W2 = (Delta_2.T @ H_1) / n_examples
    grad_B2 = Delta_2.mean(axis=0)
    
    # backprop layer 1
    S_1 = sigmoid(Z_1)
    Delta_1 = (Delta_2 @ m.W_2) * S_1 * (1 - S_1)
    grad_W1 = (Delta_1.T @ X) / n_examples
    grad_B1 = Delta_1.mean(axis=0)

    # update weights
    m.W_2 = m.W_2 - lr * grad_W2
    m.B_2 = m.B_2 - lr * grad_B2
    m.W_1 = m.W_1 - lr * grad_W1
    m.B_1 = m.B_1 - lr * grad_B1

    return (m, L)

if __name__ == "__main__":

    n_input = int(sys.argv[1])
    n_hidden = int(sys.argv[2])
    n_output = int(sys.argv[3])

    classes = 10
    lr = 0.02
    epochs = 100

    d_train = sep_labels(np.loadtxt(sys.argv[4], delimiter=',', skiprows=1), classes) # (60000, 785) first col is label
    d_test = sep_labels(np.loadtxt(sys.argv[5], delimiter=',', skiprows=1), classes) # (10000, 785) first col is label

    model = init_model(n_input, n_hidden, n_output)

    for i in range(epochs):
        loss = 0
        # mb mini batch
        for mb in batch(d_train, 32):
            model, loss = train(mb, model, lr)
        print(loss)