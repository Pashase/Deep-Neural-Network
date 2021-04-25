import numpy as np
import typing as tp


def relu(Z) -> tp.Tuple[np.ndarray, np.ndarray]:
    """

    Implements the ReLU activation in numpy

    Arguments:
    Z - represents a vector of the following structure: Z = w^T * x + b
              where w - weights, b - bias unit

    Return's:
    A - out of relu(z), same shape as Z
    Z_history (also called a cache) - returns Z as well, useful during backpropagation part

    """
    A = np.maximum(0, Z)

    assert A.shape == Z.shape

    Z_history = Z

    return A, Z_history


def relu_backward(dA: np.array, activation_cache: np.array) -> np.ndarray:
    """

    Implement the backward propagation for a single ReLU layer.

    Arguments:
    dA -- post-activation gradient, of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Return's:
    dZ -- Gradient of the cost with respect to Z

    """
    Z = activation_cache

    dZ = np.array(dA, copy=True)

    # if z <= 0, you should set dZ to 0
    dZ[Z <= 0] = 0

    assert dZ.shape == Z.shape

    return dZ
