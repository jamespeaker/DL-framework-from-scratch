import numpy as np


class MNIST_preprocessed():
    """
    A class to allow an easy import of preprocessed MNIST data.

    Preprocessing involved
     - scaling the pixel values to [0,1].
     - flattening the 28x28 images to 784 arrays.
     - one hot encoding the targets.
    """

    def fetch_data(self):
        """
        Function to return the preprocessed MNIST dataset.
        """

        with open('Data\mnist_X_train.npy', 'rb') as f:
            X_train = np.load(f)

        with open('Data\mnist_y_train.npy', 'rb') as f:
            y_train = np.load(f)

        with open('Data\mnist_X_test.npy', 'rb') as f:
            X_test = np.load(f)

        with open('Data\mnist_y_test.npy', 'rb') as f:
            y_test = np.load(f)

        return [[X_train, y_train], [X_test, y_test]]