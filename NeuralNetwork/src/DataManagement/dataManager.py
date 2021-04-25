import functools
import h5py

import typing as tp
import numpy as np
import pandas as pd

import src.DataManagement.paths as __paths__

train_data_dict = tp.Dict[str, np.ndarray]
test_data_dict = tp.Dict[str, np.ndarray]

# --
TRAINING_MNIST_DATA_LOCAL = __paths__.TRAINING_MNIST_DATA_LOCAL
TEST_MNIST_DATA_LOCAL = __paths__.TEST_MNIST_DATA_LOCAL

TRAINING_CAT_NON_CAT_DATA_LOCAL = __paths__.TRAINING_CAT_NON_CAT_DATA_LOCAL
TEST_CAT_NON_CAT_DATA_LOCAL = __paths__.TEST_CAT_NON_CAT_DATA_LOCAL


# --

# def get_local_data_path(project_path: str, problem_name: str):
#     file_names_list = [os.path.join(dir_name, file_name)
#                        for dir_name, _, file_names in os.walk(project_path)
#                        for file_name in file_names if problem_name + '/Data' in dir_name and '.csv' in file_name]
#     return file_names_list
#
#
# def get_project_abs_path() -> str:
#     return os.path.dirname(os.path.abspath('__main__'))

def get_MNIST_Data(train_data_path: str = None, test_data_path: str = None) -> tp.Tuple[train_data_dict, test_data_dict]:
    """
        Input arguments:
         train_data_path, test_data_path -- path's to data in csv format

         Return's:
         train_data, test_data -- python dictionaries which consists each data respectively
    """
    train_data_path = train_data_path if train_data_path is not None else TRAINING_MNIST_DATA_LOCAL
    test_data_path = test_data_path if test_data_path is not None else TEST_MNIST_DATA_LOCAL

    train_data_dataframe = pd.read_csv(train_data_path)
    test_data_dataframe = pd.read_csv(test_data_path)

    X_train = train_data_dataframe.drop(['label'], axis=1).to_numpy().T
    y_train = train_data_dataframe['label'].to_numpy()

    X_test = test_data_dataframe.drop(['label'], axis=1).to_numpy().T
    y_test = test_data_dataframe['label'].to_numpy()

    train_data = {'X': X_train, 'y': np.array([y_train])}
    test_data = {'X': X_test, 'y': np.array([y_test])}

    return train_data, test_data


# get CatNonCAt data and preprocessing it

def standardize_data(image2vector_function: tp.Callable):
    @functools.wraps(image2vector_function)
    def wrapper(*args, **kwargs):
        X_train_flatten, X_test_flatten = image2vector_function(*args, **kwargs)

        """
            To represent color images, the red, green and blue channels (RGB) must be specified for each pixel,
            and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
        """

        # standardize dataset
        X_train_flatten /= 255.
        X_test_flatten /= 255.

        return X_train_flatten, X_test_flatten

    return wrapper


@standardize_data
def image2vector(X_train: np.array, X_test: np.array) -> tp.Tuple[np.ndarray, np.ndarray]:
    m_train, height, width, depth = X_train.shape
    n_x = height * width * depth

    m_test = len(X_test)

    X_train_flatten = X_train.reshape(n_x, m_train)
    X_test_flatten = X_test.reshape(n_x, m_test)

    return X_train_flatten, X_test_flatten


def preprocessing_CatNonCat_data(get_data_function: tp.Callable):
    @functools.wraps(get_data_function)
    def wrapper(*args, **kwargs):
        X_train, y_train, X_test, y_test, classes = get_data_function(*args, **kwargs)

        # Let's present our test and training images as vectors (r_x as count features in each vector = R * G * B * 3)
        X_train_flatten, X_test_flatten = image2vector(X_train, X_test)

        train_data = {'X': X_train_flatten, 'y': np.array([y_train])}
        test_data = {'X': X_test_flatten, 'y': np.array([y_test])}

        return train_data, test_data, classes

    return wrapper


@preprocessing_CatNonCat_data
def get_CatNonCat_Data(train_data_path: str = None, test_data_path: str = None):
    """
        Input arguments:
        train_data_path -- path to training data in .h5 format
        test_data_path -- path to test data in .h5 format

        Return's:
        Splitted data to train/test examples

        train_set_x_orig - vector of images, each image represents as a vector of size (64*64*3, 1)
        train_set_y_orig - vector correct answers (cat/noncat), of a size (m, 1), m - samples count in dataset

        test_set_x_orig - vector of images, each image represents as a vector of size (64*64*3, 1)
        test_set_y_orig - vector of correct answers (cat/noncat)
        """
    train_data_path = TRAINING_CAT_NON_CAT_DATA_LOCAL if train_data_path is None else train_data_path
    test_data_path = TEST_CAT_NON_CAT_DATA_LOCAL if test_data_path is None else test_data_path

    with h5py.File(train_data_path, 'r') as train_dataset, h5py.File(test_data_path, 'r') as test_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]).astype('float')
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

        test_set_x_orig = np.array(test_dataset["test_set_x"][:]).astype('float')
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def get_binary_matrix(y: np.array, *, num_of_labels=10) -> np.ndarray:
    """
    Convert each y_i-label to a (num_of_labels, 1) column vector and return a matrix of this vectors

    Parameters:
    y -- true column vector of labels
    -->
    If we choose a Multiclassification problem, y has next format [[1,3,5,3,4,7, ...]]
    Binary classification problem - [[0,1,0,0,1,0, ...]]
    -->

    problem_name -- name of the chosen problem
    num_of_labels -- count of classes in a multi-classification task

    Return's:
    binary_matrix -- matrix of converted y_i to vector labels

    """

    # We need to transform input y to a column vector (according to a chosen problem)
    y_preprocessed = np.array([np.squeeze(y)])
    n_y, m_samples = y_preprocessed.shape

    # check: y is a vector
    assert n_y == 1

    binary_matrix = np.zeros((num_of_labels, m_samples))
    index_array = [(index, position) for index, position in enumerate(np.squeeze(y))]

    col_index, row_index = list(zip(*index_array))
    binary_matrix[row_index, col_index] = 1.0

    return binary_matrix
