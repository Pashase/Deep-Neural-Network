import numpy as np
import pandas as pd
import typing as tp
import seaborn as sns

import matplotlib.pyplot as plt

import src.Labels.label as __label__

import src.NeuralNetwork.Architecture as Architecture
import src.NeuralNetwork.Cost as __cost__
import src.NeuralNetwork.propogations as __propogations__

NetworkArchitecture = Architecture.NetworkArchitecture

# -- optimization algorithms
GRADIENT_DESCENT_OPTIMIZATION = 'gradient_descent'
BATCH_GRADIENT_DESCENT_OPTIMIZATION = 'batch_gradient_descent'
STOCHASTIC_GRADIENT_DESCENT_OPTIMIZATION = 'stochastic_gradient_descent'

# -- overfitting prevention algorithms
REGULARIZATION_METHOD = 'regularization'
DROPOUT_METHOD = 'dropout'


def initialize_weights(nn_architecture: NetworkArchitecture,
                       *,
                       small_parameter=0.01,
                       seed_number: int = 42) -> tp.Dict[str, tp.Union[float, np.ndarray]]:
    np.random.seed(seed_number)

    weights = dict()
    label_W, label_b, _, label_A = __label__.Label.values()

    for indexLayer, layer in enumerate(nn_architecture.Layers, start=1):
        # имеем дело со входным слоем
        weights[label_W + str(indexLayer)] = np.random.randn(layer.neuronsCount,
                                                             layer.prevLayerNeuronsCount) * small_parameter
        weights[label_b + str(indexLayer)] = np.zeros((layer.neuronsCount, 1))

        assert weights[label_W + str(indexLayer)].shape == (layer.neuronsCount, layer.prevLayerNeuronsCount)
        assert weights[label_b + str(indexLayer)].shape == (layer.neuronsCount, 1)

    return weights


def gradientDescent(weights: dict,
                    gradients: dict,
                    alpha: float,
                    nn_architecture: NetworkArchitecture) -> tp.Dict[str, tp.Union[float, np.ndarray]]:
    """

    Update parameters using simple gradient descent

    Arguments:
    weights -- dictionary containing your parameters
    gradients -- dictionary containing your gradients, output of full_L_backward_propogation
    alpha -- learning rate for gradient descent
    nn_architecture -- nn model architecture

    Return's:
    weights -- dictionary that containing updated weights after each iteration of gd
                  weights["W" + str(iLayer)] = ...
                  weights["b" + str(iLayer)] = ...

    """
    label_W, label_b, *_ = __label__.Label.values()

    for indexLayer, layer in enumerate(nn_architecture.Layers, start=1):
        weights[label_W + str(indexLayer)] -= alpha * gradients['dW' + str(indexLayer)]
        weights[label_b + str(indexLayer)] -= alpha * gradients['db' + str(indexLayer)]

    return weights


class Model(NetworkArchitecture):
    def __init__(self, layers: tp.List[Architecture.Layer], data: dict, weights: dict):
        # small_parameter used as shift of init weights in initialize_weights
        super().__init__(layers)
        self.data = data
        # here, each instance of a Model class will have init with random weights
        # self.weights = initialize_weights(NetworkArchitecture(layers), small_parameter=small_param)
        self.weights = weights

    def start_learning(self, *, alpha: float = 0.003, optimization_algorithm_name: str,
                       countIterations: int, print_cost=True):

        optimization_algorithm = gradientDescent

        costs = dict()
        y = self.data['y']

        nn_architecture = NetworkArchitecture(self.Layers)

        if optimization_algorithm_name == GRADIENT_DESCENT_OPTIMIZATION:
            optimization_algorithm = gradientDescent
        elif optimization_algorithm_name == BATCH_GRADIENT_DESCENT_OPTIMIZATION:
            raise NotImplementedError(f'{optimization_algorithm_name} is not implemented yet')
        elif optimization_algorithm_name == STOCHASTIC_GRADIENT_DESCENT_OPTIMIZATION:
            raise NotImplementedError(f'{optimization_algorithm_name} is not implemented yet')

        # Initialize parameters
        initialized_weights = self.weights

        for iteration in range(1, countIterations + 1):
            # Forward propagation
            AL, caches = __propogations__.full_L_forward_propogation(nn_architecture, self.data, initialized_weights)

            # Calculate a cost
            label_count = nn_architecture.Layers[-1].neuronsCount
            cost_value = __cost__.Cost(AL, y, label_count)
            costs[iteration] = cost_value

            # Backward propagation
            gradients = __propogations__.full_L_backward_propogation(y, AL, caches, nn_architecture)

            # Update weights using gradient descent(i'll implement another algorithms, that minimize a cost later)
            initialized_weights = optimization_algorithm(initialized_weights, gradients,
                                                         alpha, nn_architecture)

            # Print the cost every 100 training example
            if print_cost and iteration % 3 == 0:
                print(f'Cost value after {iteration} iteration is {cost_value}')

        return initialized_weights, costs

    def predict(self, test_data: dict):
        weights = self.weights

        # AL -- predictions
        AL, caches = __propogations__.full_L_forward_propogation(NetworkArchitecture(self.Layers), test_data, weights)

        predictions, caches = np.array([np.where(prediction_vector > 0.5, 1, 0) for prediction_vector in AL])

        return predictions, caches

    def __str__(self) -> str:
        return f'-------------------------- Model information --------------------------\n' \
               f'Layers\nCount layers in model: {self.LayersCount}\n' \
               f'Layers:\n{self.Layers}\n' \
               f'Final trained weights:\n{self.weights}'

    @staticmethod
    def build_plot_convergence(convergence_history: dict) -> None:
        sns.set(rc={'figure.figsize': (11.7, 8.27)})

        convergence_data = pd.DataFrame({'Iterations': convergence_history.keys(),
                                         'Cost': convergence_history.values()})

        sns.scatterplot(x=convergence_data['Iterations'], y=convergence_data['Cost'], data=convergence_data)
        plt.show()
