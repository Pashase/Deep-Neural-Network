import src.NeuralNetwork.sigmoid as __sigmoid__
import src.NeuralNetwork.relu as __relu__

from NeuralNetwork.src.NeuralNetwork.Architecture import InputLayer
from NeuralNetwork.src.NeuralNetwork.Architecture import Layer

# Digit recognition example ini architecture
dr_layers = [
    # во входной слой обязательно передать данные (тренировочные/тестовые)
    Layer(prevLayerNeuronsCount=InputLayer(data).n_x, neuronsCount=20, activationFunction=__relu__.relu,
          backward_activation=__relu__.relu_backward),
    Layer(prevLayerNeuronsCount=20, neuronsCount=16, activationFunction=__relu__.relu,
          backward_activation=__relu__.relu_backward),
    Layer(prevLayerNeuronsCount=16, neuronsCount=13, activationFunction=__relu__.relu,
          backward_activation=__relu__.relu_backward),
    Layer(prevLayerNeuronsCount=13, neuronsCount=10, activationFunction=__relu__.relu,
          backward_activation=__relu__.relu_backward),
    Layer(prevLayerNeuronsCount=10, neuronsCount=10, activationFunction=__sigmoid__.sigmoid,
          backward_activation=__sigmoid__.sigmoid_backward),
]

# Cat non Cat
cnc_layers = [Layer(prevLayerNeuronsCount=InputLayer(data).n_x, neuronsCount=20, activationFunction=__relu__.relu,
                    backward_activation=__relu__.relu_backward),
              Layer(prevLayerNeuronsCount=20, neuronsCount=16, activationFunction=__relu__.relu,
                    backward_activation=__relu__.relu_backward),
              Layer(prevLayerNeuronsCount=16, neuronsCount=10, activationFunction=__relu__.relu,
                    backward_activation=__relu__.relu_backward),
              Layer(prevLayerNeuronsCount=10, neuronsCount=2, activationFunction=__sigmoid__.sigmoid,
                    backward_activation=__sigmoid__.sigmoid_backward),
              ]
