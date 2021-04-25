import numpy as np


def tanh(Z):
    """

    Implements the tanh activation in numpy

    Arguments:
    Z - represents a vector of the following structure: Z = w^T * x + b
              where w - weights, b - bias unit

    Return's:
    A - out of tanh(z), same shape as Z
    Z_history (also called a cache) - returns Z as well, useful during backpropagation part

    """

    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    assert A.shape == Z.shape

    Z_history = Z

    return A, Z_history


def tanh_backward(dA: np.array, activation_cache: np.array):
    Z = activation_cache

    sigm = 1 / (1 + np.exp(-Z))
    dZ = dA * (1 - np.power(sigm, 2))

    assert dZ.shape == Z.shape

    return dZ
