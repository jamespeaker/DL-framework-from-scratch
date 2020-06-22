import numpy as np

class Dense():
    """
    The dense layer (also known as the fully connected layer).

    This class uses its forward method to forward propagate and its backward method to
    update weights and biases b, and to back propagate a new gradient to previous layers.

    Attributes
    ----------
    W : numpy.ndarray
        the weights of the layer, used in the forward step to compute aW.T+b.
    b : numpy.ndarray
        the biases of the layer, used in the forward step to compute aW.T+b.
    num_neurons : int
        the number of neurons in this layer.
    input shape : int
        length of input array.
    input : numpy.ndarray
        variable assigned in forward() to store the layer input to use in backward() when
        calculating gradients.
    """

    def __init__(self, num_neurons, input_shape=None):
        """
        Instantiates a dense layer with num_neurons neurons. If the input shape is given
        then the method calls compile_layer to instantiate the weights W and biases b.

        :param num_neurons: number of neurons in this layer.
        :param input_shape: scalar size of the input. If this is the first layer of the
          network the input is x. Otherwise the input is the output of the previous layer.
        """

        self.num_neurons = num_neurons

        if input_shape is not None:
            self.compile_layer(input_shape)

    def compile_layer(self, input_shape):
        """
        Instantiates the weights W and biases b, to be used in the forward method.

        :param input_shape: scalar size of the input to the layer.
        """

        self.input_shape = input_shape
        self.W = np.random.normal(0, 0.01, (self.num_neurons, self.input_shape))
        self.b = np.random.normal(0, 0.01, self.num_neurons)

    def forward(self, a):
        """
        Performs the forward pass of the layer. This method also saves the input 'a' as
        self.input for future use when calculating gradients. These gradients are  used
        to update the weights and biases in the layer.

        :param a: input to the layer.
        :return: the num_neurons-sized vector aW.T + b, containing the weighted sums.
        """

        self.input = a
        weighted_sums = np.matmul(a, self.W.T) + self.b

        return weighted_sums

    def backward(self, gradient_backpropagated, learning_rate):
        """
        Performs the backward pass of the layer. This includes updating weights W,
        updating biases b and returning the new gradients to be backpropagated
        to the previous layer.

        :param gradient_backpropagated: gradient being 'backpropagated' through the
        network. This is essentially the product of the chain of local gradients
        calculated in following layers. These local gradients are the gradient of the
        layer output with respect to the layer input. When we take the product of
        these (and the gradient of the loss), we get the gradient of the loss with
        respect to the input.
        :param learning_rate: learning rate of the model, assigned when model is compiled.
        :return: the gradient to be 'backpropagated' to the previous layer.
        """

        self.update_weights(gradient_backpropagated, learning_rate)
        self.update_biases(gradient_backpropagated, learning_rate)

        new_gradient_to_be_backpropagated = self.update_gradient_to_be_backpropagated(gradient_backpropagated)

        return new_gradient_to_be_backpropagated

    def update_biases(self, gradient_backpropagated, learning_rate):
        """
        Update the layer biases b.

        :param gradient_backpropagated: gradient being 'backpropagated' through the network.
        :param learning_rate: learning rate of the model, assigned when model is compiled.
        """

        self.b = self.b - learning_rate * gradient_backpropagated

    def update_weights(self, gradient_backpropagated, learning_rate):
        """
        Update the layer weights W. First the weight gradients are calculated and then these
        are used to update the weights.

        :param gradient_backpropagated: gradient being 'backpropagated' through the network.
        :param learning_rate: learning rate of the model, assigned when model is compiled.
        """

        weight_gradients = self.calculate_weight_gradients(gradient_backpropagated)
        self.W = self.W - learning_rate * weight_gradients


    def calculate_weight_gradients(self, gradient_backpropagated):
        """
        Calculates the gradient of the loss with respect to the weights. To do this we must
        multiply gradient_backpropagated by the gradient of the output with respect to the
        weights. The gradient of the output aW.T + b with respect to W is 'a'. We use the
        outer product which makes sense as we must have weight_gradients.shape == W.shape.

        :param gradient_backpropagated: gradient being 'backpropagated' through the network.
        :return: the gradient of the loss with respect to the weights W.
        """

        return np.squeeze(np.multiply.outer(gradient_backpropagated, self.input))


    def update_gradient_to_be_backpropagated(self, gradient_backpropagated):
        """
        Update the gradient to be 'backpropagated' to the previous layer. To do this we must
        multiply gradient_backpropagated by the local gradient of this layer. The local
        gradient is the gradient of the output with respect to the input. In this case, the
        we have input 'a' and output aW.T + b, making our local gradient W. Therefore our
        updated gradient to be backpropagated is the matrix multiplication of
        gradient_backpropagated and W.

        :param gradient_backpropagated: gradient being 'backpropagated' through the network.
        :return:the gradient of the loss with respect to the input a.
        """

        return np.matmul(gradient_backpropagated, self.W)