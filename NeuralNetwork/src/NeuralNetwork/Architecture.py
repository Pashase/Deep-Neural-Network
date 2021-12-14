import typing as tp

import src.NeuralNetwork.sigmoid as __sigmoid__
import src.NeuralNetwork.relu as __relu__

from src.DataManagement.Problems.default_architectures import dr_layers
from src.DataManagement.Problems.default_architectures import cnc_layers

from src.DataManagement.Problems.tasks import ConsiderTasks
from matplotlib import pyplot as plt


class Layer(tp.NamedTuple):
    prevLayerNeuronsCount: tp.Union[int, None]
    neuronsCount: int
    activationFunction: tp.Callable
    backward_activation: tp.Callable


class InputLayer(object):
    def __init__(self, data: dict):
        count_features, m_samples = data['X'].shape

        self.data = data
        self.n_x = count_features
        self.m = m_samples


class NetworkArchitecture(object):
    def __init__(self, layers: tp.List[Layer]):
        self.Layers = layers
        self.LayersCount = len(layers)

    @staticmethod
    def init_NN_architecture(problem_name: str) -> tp.List[Layer]:
        layers = None
        # this method could be edited according to a new task
        if problem_name == ConsiderTasks.DigitRecognitionTask.value:
            layers = dr_layers
        elif problem_name == ConsiderTasks.CatNonCatTask.value:
            layers = cnc_layers
        else:
            raise ValueError(f'There is no task {problem_name}!')

        return layers

    def draw_neural_net(self, ax, left, right, bottom, top) -> None:
        """
        Draw a neural network using matplotlib

        :usage:
            >>> fig = plt.figure(figsize=(12, 12))
            >>> draw_neural_net(fig.gca(), .1, .9, .1, .9)

        :parameters:
            - ax : matplotlib.axes.AxesSubplot
                The axes on which to plot the cartoon (get e.g. by plt.gca())
            - left : float
                The center of the leftmost node(s) will be placed here
            - right : float
                The center of the rightmost node(s) will be placed here
            - bottom : float
                The center of the bottommost node(s) will be placed here
            - top : float
                The center of the topmost node(s) will be placed here
            - layer_sizes : list of int
                List of layer sizes, including input and output dimensionality
        """
        layer_sizes = [layer.neuronsCount for layer in self.Layers]

        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)

        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size):
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)

        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                    ax.add_artist(line)

        return None
