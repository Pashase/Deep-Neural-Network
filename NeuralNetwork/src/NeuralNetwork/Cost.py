import numpy as np
from src.DataManagement.dataManager import get_binary_matrix


def CostFunction(AL: np.array, y: np.array, labels_count: int) -> float:
    y_binary = get_binary_matrix(y, num_of_labels=labels_count)

    _, m = y_binary.shape

    # logprobs = np.multiply(-y_binary, np.log(AL)) - np.multiply((1 - y_binary), np.log(1 - AL))
    # cost_value = 1 / m * np.sum(logprobs)

    # cost_value = -1/m * (np.dot(y_binary, np.log(AL.T)) + np.dot(1 - y, np.log(1 - AL).T))
    cost_value = -1/m * np.sum(y_binary * np.log(AL) + (1 - y_binary) * (np.log(1 - AL)))

    # To make sure your cost's shape is what we expect (e.g. this turns [[10]] into 10)
    cost_value = float(np.squeeze(cost_value))

    return cost_value


def Cost(AL: np.array, y: np.array, labels_count: int) -> float:
    """
    Implement's the Cost function.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    y, shape of (1, number of examples)
    labels_count -- count of labels, need to know what task are we dealing with -> how to generate a binary_matrix

    Returns:
    cost_value -- cross-entropy cost

    """
    cost_value = CostFunction(AL, y, labels_count)

    return cost_value
