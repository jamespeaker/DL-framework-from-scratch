from Losses import Categorical_crossentropy
import numpy as np
import math

class Model():
    """
    The neural network. This class compiles a neural network based on the list of layers/activations
    given in the constructor. E.g. neural_net = Model([Dense(10),Dense(10),Softmax()]). The class
    uses forward_pass() to iterate forward through the layers/activations to predict. It uses
    backward_pass() to iterate backwards to provide gradients to allow updates to the weights and
    biases in the dense layers.

    Attributes
    ----------
    layers : list
        List of the layers or activations in the model. E.g. [Dense(10), Dense(20), Softmax()].
    loss : object from Losses.py
        The loss used in the model to determine performance.
    learning_rate : float
        Rate to update the parameters with update rule: param = param - learning_rate*gradient.
    """

    def __init__(self, layer_list):
        """
        Assigns the layer_list to the field layers.

        :param layer_list: a list containing only layers or activations.
        """
        self.layers = layer_list

    def compile_model(self, loss_string, learning_rate):
        """
        Compiles the model. First it uses the loss_string to instantiate a loss object and assigns
        this to the attribute loss. Then it uses the layer/activation method compile_layer() to
        assign the input shapes, which are the previous layer's number of neurons.

        :param loss_string: the string describing the loss e.g. "MSE" or "categorical_crossentropy".
        :param learning_rate: rate at which parameters are adjusted, i.e. param=param-gradient*lr.
        """

        if loss_string == "MSE":
            self.loss = Mean_squared_error()
        elif loss_string == "categorical_crossentropy":
            self.loss = Categorical_crossentropy()

        for i in range(len(self.layers) - 1):
            self.layers[i + 1].compile_layer(input_shape=self.layers[i].num_neurons)

        self.learning_rate = learning_rate

    def train_on_batch(self, x, target):
        """
        The model updates its parameters using a batch of data. This batch provides an estimate
        of the gradient of the loss. This gradient is used to update the parameters in the method
        backward_pass(). Firstly, the model does a forward pass using the batch. This forward pass
        is necessary to compute the gradients. Secondly, backward_pass() is called to update the
        parameters.

        :param x: batch of feature data to train on model.
        :param target: batch of target data to train on model.
        """

        NN_output, _ = self.forward_pass(x, target)

        self.backward_pass(NN_output, target)

    def forward_pass(self, x, target):
        """
        Forward pass through the model, using x. This method also calculates the loss for the
        particular example given.

        :param x: batch of feature data to train on model.
        :param target: batch of target data to train on model.
        :return: the output and loss of the neural network for this particular example.
        """

        for layer in self.layers:
            x = layer.forward(x)

        NN_output = x

        loss = self.loss.calculate_loss(NN_output, target)

        return NN_output, loss

    def backward_pass(self, NN_output, target):
        """
        The backward pass of the network where the gradients of the loss are calculated from the
        last layer of the network to the first. This is done by first calculating the gradient
        of the loss and then backpropagating this through the layers, using the reverse of
        the attribute layers.

        :param NN_output: the output of the neural network, calculated using forward_pass().
        :param target: the target of the neural network for that example.
        """
        gradient = self.loss.calculate_gradient(NN_output, target)

        gradient = self.clip_grad(gradient)

        self.layers.reverse()

        for layer in self.layers:
            gradient = layer.backward(gradient, self.learning_rate)
            gradient = self.clip_grad(gradient)
        self.layers.reverse()


    def predict(self, x):
        """
        Give the model prediction using feature x.

        :param x: feature data.
        :return: neural network prediction given feature x.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def clip_grad(self, grad_array, value=0.5):
        """
        Clips the gradient to avoid the exploding gradient problem. Method iterates through the
        gradients and if is above/below +-value, it is clipped to +-value. Furthermore, if a
        value is NaN, this is clipped to value.

        :param grad_array: gradient being backpropagated through the network.
        :param value: value to clip grad to. Large positive values are clipped to value, large
        negative values are clipped to -value.
        :return: the clipped gradient array.
        """

        if grad_array.ndim == 2:
            grad_array = np.squeeze(grad_array)

        for i in range(grad_array.shape[0]):
            if grad_array[i] < -value:
                grad_array[i] = -value
            elif grad_array[i] > value:
                grad_array[i] = value
            elif math.isnan(grad_array[i]):
                grad_array[i] = value

        return grad_array




