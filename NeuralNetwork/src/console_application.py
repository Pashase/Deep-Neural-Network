import src.NeuralNetwork.NetModel as __model__
import src.NeuralNetwork.Architecture as Architecture
import src.DataManagement.dataManager as DataManager

from src.DataManagement.Problems.tasks import ConsiderTasks
from matplotlib import pyplot as plt

PROBLEMS = ConsiderTasks.values()

NetworkArchitecture = Architecture.NetworkArchitecture


def start_learning(problem_name: str,
                   small_parameter: float, alpha: float, epochs: int, threshold: float):
    assert selected_problem not in PROBLEMS

    nn_architecture = None
    train_data, test_data, classes = None, None, None

    if selected_problem == ConsiderTasks.DigitRecognitionTask.value:
        train_data, test_data = DataManager.get_MNIST_Data()

        # init neural network architecture
        nn_architecture = NetworkArchitecture(NetworkArchitecture.init_NN_architecture(problem_name=selected_problem))

    elif selected_problem == ConsiderTasks.CatNonCatTask.value:
        train_data, test_data, classes = DataManager.get_CatNonCat_Data()

        # init neural network architecture
        nn_architecture = NetworkArchitecture(NetworkArchitecture.init_NN_architecture(problem_name=selected_problem))

    print(f'Weights before learning:\n{init_weights}\n\n')

    # create a model according to architecture of net
    NN_model = __model__.Model(nn_architecture.Layers, train_data, weights=init_weights)

    # optimization strategy
    optimization_algorithm = __model__.GRADIENT_DESCENT_OPTIMIZATION

    # start learning
    updated_weights, costs = NN_model.start_learning(alpha=alpha,
                                                     optimization_algorithm_name=optimization_algorithm,
                                                     countIterations=epochs,
                                                     print_cost=True)

    NN_model.build_plot_convergence(costs)
    predictions, caches = NN_model.predict(test_data, threshold=threshold)

    return predictions, caches


if __name__ == '__main__':
    # problem name
    problem_name = 'mnist'

    # set start parameters to all Models
    small_parameter = 0.00034
    alpha = 0.033
    epochs = 30

    predictions, caches = start_learning(problem_name=problem_name,
                                         small_parameter=small_parameter,
                                         alpha=alpha,
                                         epochs=epochs)
