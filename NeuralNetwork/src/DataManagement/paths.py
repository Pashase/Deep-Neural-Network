import os


def get_local_path(path: str) -> str:
    return os.path.dirname(os.path.abspath(__file__)) + path


# get MNIST data from my google drive (non-local)
TRAINING_MNIST_DATA = 'https://drive.google.com/file/d/1hVLbPa_odayF2g-Nsjfd7weI2Cy2DQwW/view?usp=sharing'
TEST_MNIST_DATA = 'https://drive.google.com/file/d/1vz02c_roOBZ_d3qPksVvqsxCAZSa2ovR/view?usp=sharing'

# get local MNIST data local (data should be in /Problems/... folder)
TRAINING_MNIST_DATA_LOCAL = '/Problems/DigitRecognition/Data/mnist_train.csv'
TEST_MNIST_DATA_LOCAL = '/Problems/DigitRecognition/Data/mnist_test.csv'

TRAINING_MNIST_DATA_LOCAL = get_local_path(TRAINING_MNIST_DATA_LOCAL)
TEST_MNIST_DATA_LOCAL = get_local_path(TEST_MNIST_DATA_LOCAL)

# CatNonCat data is used only in local format
TRAINING_CAT_NON_CAT_DATA_LOCAL = '/Problems/CatNonCat/Data/train_catvnoncat.h5'
TEST_CAT_NON_CAT_DATA_LOCAL = '/Problems/CatNonCat/Data/test_catvnoncat.h5'

TRAINING_CAT_NON_CAT_DATA_LOCAL = get_local_path(TRAINING_CAT_NON_CAT_DATA_LOCAL)
TEST_CAT_NON_CAT_DATA_LOCAL = get_local_path(TEST_CAT_NON_CAT_DATA_LOCAL)
