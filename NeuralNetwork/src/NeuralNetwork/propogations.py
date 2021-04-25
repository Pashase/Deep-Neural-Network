import numpy as np
import typing as tp

import src.DataManagement.dataManager as DataManager
import src.NeuralNetwork.Architecture as Architecture
import src.Labels.label as __label__

Label = __label__.Label
get_binary_matrix = DataManager.get_binary_matrix
NetworkArchitecture = Architecture.NetworkArchitecture


def forward_propogation_step(A_prev: np.array, W: np.array, b: np.array, activationFunction: tp.Callable):
    """
    Implement a step of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A_prev) + b

    assert Z.shape == (W.shape[0], A_prev.shape[1])

    linear_cache = (A_prev, W, b)

    A, activation_cache = activationFunction(Z)
    assert A.shape == (W.shape[0], A_prev.shape[1])

    all_cache = (linear_cache, activation_cache)

    return A, all_cache


def full_L_forward_propogation(nn_architecture: NetworkArchitecture, data: dict, weights: dict):
    """
    Implement forward propagation.

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Return's:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_..._forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    X = data['X']

    label_W, label_b, *_ = Label.values()
    n_x, m_samples = X.shape

    caches = []

    # contains all post-activation values from all layers
    A_all = []

    # A0 == X
    A = X
    A_all.append(A)

    for indexLayer, layer in enumerate(nn_architecture.Layers, start=1):
        A_previous = A

        current_W, current_b = weights[label_W + str(indexLayer)], weights[label_b + str(indexLayer)]
        A, cache = forward_propogation_step(A_previous, current_W, current_b, layer.activationFunction)

        caches.append(cache)
        A_all.append(A)

    # AL is the last value of A_all
    AL = A_all[nn_architecture.LayersCount]

    n_last = nn_architecture.Layers[-1].neuronsCount
    assert AL.shape == (n_last, m_samples)

    return AL, caches


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dA_prev, dW, db


def backward_propogation_step(dA: np.array, cache: list, backward_activationFunction: tp.Callable):
    linear_cache, activation_cache = cache

    dZ = backward_activationFunction(dA, activation_cache)
    dA_previous, dW, db = linear_backward(dZ, linear_cache)

    return dA_previous, dW, db


def full_L_backward_propogation(y: np.array, AL: np.array, caches: list,
                                nn_architecture: NetworkArchitecture):
    gradients = dict()

    L = nn_architecture.LayersCount

    # should know, how to generate binary matrix from y (shape)
    number_of_labels = nn_architecture.Layers[-1].neuronsCount
    y = get_binary_matrix(y, num_of_labels=number_of_labels)

    # Initializing the backpropagation
    dAL = - (np.divide(y, AL) - np.divide(1 - y, 1 - AL))

    # For last layer
    current_cache = caches[L - 1]
    gradients['dA' + str(L)], \
    gradients['dW' + str(L)], \
    gradients['db' + str(L)] = backward_propogation_step(dAL, current_cache,
                                                         nn_architecture.Layers[-1].backward_activation)

    for previousIndexLayer, layer in reversed(list(enumerate(nn_architecture.Layers[:L - 1], start=1))):
        current_cache = caches[previousIndexLayer - 1]

        dA_prev_temp, dW_temp, db_temp = backward_propogation_step(gradients["dA" + str(previousIndexLayer + 1)],
                                                                   current_cache, layer.backward_activation)

        gradients["dA" + str(previousIndexLayer)] = dA_prev_temp
        gradients["dW" + str(previousIndexLayer)] = dW_temp
        gradients["db" + str(previousIndexLayer)] = db_temp

    return gradients
