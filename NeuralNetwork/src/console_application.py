import src.NeuralNetwork.NetModel as __model__
import src.NeuralNetwork.Architecture as Architecture
import src.DataManagement.dataManager as DataManager

from src.DataManagement.Problems.tasks import ConsiderTasks
from matplotlib import pyplot as plt

PROBLEMS = ConsiderTasks.values()

NetworkArchitecture = Architecture.NetworkArchitecture


def main(small_parameter: float, alpha: float, epochs: int) -> None:
    selected_problem = input(f'Choose a problem from {PROBLEMS}\n')

    if selected_problem not in PROBLEMS:
        raise OSError('Illegal argument')

    # According to selected problem
    if selected_problem == ConsiderTasks.DigitRecognitionTask.value:
        # get MNIST data
        train_data, test_data = DataManager.get_MNIST_Data()

        # init neural network architecture
        nn_architecture = NetworkArchitecture(NetworkArchitecture.init_NN_architecture(train_data,
                                                                                       problem_name=selected_problem))

        # draw a neural net
        # fig = plt.figure(figsize=(8, 8))
        # nn_architecture.draw_neural_net(fig.gca(), .1, .9, .1, .9)
        # plt.show()

        # set start parameters
        # small_parameter = 0.00034
        # alpha = 0.033
        # epochs = 15

        # init weights
        init_weights = __model__.initialize_weights(nn_architecture, small_parameter=small_parameter)
        print(f'Weights before learning:\n{init_weights}\n\n')

        # create a model according to architecture of net
        NN_model = __model__.Model(nn_architecture.Layers, train_data, weights=init_weights)

        # optimization
        optimization_algorithm = __model__.GRADIENT_DESCENT_OPTIMIZATION

        # start learning
        updated_weights, costs = NN_model.start_learning(alpha=alpha,
                                                         optimization_algorithm_name=optimization_algorithm,
                                                         countIterations=epochs,
                                                         print_cost=True)

        # NN_model.build_plot_convergence(costs)
        print(f'Weights after learning from out function variable:\n{updated_weights}\n\n')
        print(f'Weights after learning from Class variable:\n{updated_weights}\n\n')

    if selected_problem == ConsiderTasks.CatNonCatTask.value:
        train_data, test_data, classes = DataManager.get_CatNonCat_Data()

        nn_architecture = NetworkArchitecture(NetworkArchitecture.init_NN_architecture(train_data,
                                                                                       problem_name=selected_problem))

        fig = plt.figure(figsize=(8, 8))
        nn_architecture.draw_neural_net(fig.gca(), .1, .9, .1, .9)
        plt.show()

        # set start parameters
        # small_parameter = 0.00034
        # alpha = 0.009
        # epochs = 10000

        init_weights = __model__.initialize_weights(nn_architecture, small_parameter=small_parameter)

        # create a model according to architecture of net
        NN_model = __model__.Model(nn_architecture.Layers, train_data, weights=init_weights)

        # optimization
        optimization_algorithm = __model__.GRADIENT_DESCENT_OPTIMIZATION

        # start learning
        updated_weights, costs = NN_model.start_learning(alpha=alpha,
                                                         optimization_algorithm_name=optimization_algorithm,
                                                         countIterations=epochs,
                                                         print_cost=True)

        NN_model.build_plot_convergence(costs)

        # predictions, y_test = NN_model.predict(test_data)
        # print(f'My predictions: {predictions}\n\n'
        #       f'Real y: {y_test}')


if __name__ == '__main__':
    # set start parameters to all Models
    small_parameter = 0.00034
    alpha = 0.033
    epochs = 15

    main(small_parameter, alpha, epochs)
