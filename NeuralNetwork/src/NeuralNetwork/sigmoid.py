import numpy as np
import typing as tp


def sigmoid(Z) -> tp.Tuple[np.ndarray, np.ndarray]:
    """

    Implements the sigmoid activation in numpy

    Arguments:
    Z - represents a vector of the following structure: Z = w^T * x + b
              where w - weights, b - bias unit

    Return's:
    A - out of sigmoid(z), same shape as Z
    Z_history (also called a cache) - returns Z as well, useful during backpropogation part

    """
    A = 1 / (1 + np.exp(-Z))

    assert A.shape == Z.shape

    Z_history = Z

    return A, Z_history


def sigmoid_backward(dA: np.array, activation_cache: np.array) -> np.ndarray:
    """

    Implements the backward propagation for a single SIGMOID layer.

    Arguments:
    dA - post-activation gradient, of any shape
    activation_cache - 'Z' where we store for computing backward propagation efficiently

    Return's:
    dZ - Gradient of the cost with respect to Z

    """

    Z = activation_cache

    sigm = 1 / (1 + np.exp(-Z))
    dZ = dA * sigm * (1 - sigm)

    assert dZ.shape == Z.shape

    return dZ
