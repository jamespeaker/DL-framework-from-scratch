import numpy as np

class Softmax():
    """
    The softmax activation. This uses its forward method to compute the softmax function and uses
    its backward method to compute the gradient to backpropagate to the previous layer.

    Attributes
    ----------

    num_neurons : int
        the number of neurons in this activation.
    input shape : int
        length of input array.
    input : numpy.ndarray
        variable assigned in forward() to store the activation input to use in backward() when
        calculating gradients.
    """

    def compile_layer(self, input_shape):
        self.num_neurons = input_shape

    def forward(self, h):
        """
        Computes the softmax function of h. Method also stores self.input for usage in
        calculate_jacobian_matrix().

        :param h: the input to the activation
        :return: the softmax function of h.
        """
        self.input = h

        return np.exp(h) / (np.sum(np.exp(h))+0.01)

    def backward(self, gradient_backpropagated, learning_rate=None):
        """
        Computes the gradient to be backpropagated to the previous layer. It first calculates
        the Jacobian matrix and then takes the dot product of this with the gradient
        backpropagated from the following layer. This result is the updated gradient to
        backpropagate.

        :param gradient_backpropagated: gradient being backpropagated through the network.
        :param learning_rate: unused but required so Softmax can be treated as a layer when
        iterating through the layers with code layer.backward(grad,learning_rate=lr).
        """

        jacobian_matrix = self.calculate_jacobian_matrix()

        return np.dot(gradient_backpropagated, jacobian_matrix)

    def calculate_jacobian_matrix(self):
        """
        Calculates the Jacobian matrix of f = Softmax(h) with respect to the layer input h.
        This is a matrix where element (i,j) is the the partial derivative of output i with
        respect to input j, for i,j = 1,2,...,k, where k is the length of the input.

        When i == j, the partial derivative is
        exp(hi+hj)/(exp(h1)+...+exp(hk))^2.

        When i != j , the partial derivative is
        exp(hi) * (exp(h1)+...+exp(h(i-1))+exp(h(i+1))+...+exp(hk)) / (exp(h1)+...+exp(hk))^2.

        :return: the Jacobian matrix J(h).
        """

        h = self.input

        if h.ndim == 2:
            h = np.squeeze(h)

        exp_h = np.exp(h)

        sum_of_exp_h = np.sum(exp_h)

        jacobian_matrix = np.zeros([len(h), len(h)])
        for i in range(len(h)):
            for j in range(len(h)):
                if i != j:
                    grad = -np.exp(h[i] + h[j]) / (sum_of_exp_h ** 2 + 0.0001)
                    jacobian_matrix[i, j] = grad
                elif i == j:
                    sum_of_exp_h_without_hi = sum_of_exp_h - np.exp(h[i])
                    grad = np.exp(h[i]) * sum_of_exp_h_without_hi / (sum_of_exp_h ** 2 + 0.0001)
                    jacobian_matrix[i, j] = grad

        return jacobian_matrix