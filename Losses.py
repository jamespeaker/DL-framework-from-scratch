import numpy as np

class Categorical_crossentropy():
    """
    A class to calculate the CCE (categorical cross-entropy) loss and the local gradients.
    """

    def calculate_loss(self, predictions, targets):
        """
        Calculates the loss. This is the average value (if we have multiple predictions)
        of the log of the prediction made for the correct target. I.e. for prediction
        [0.1,0.9] and target [0,1], the loss is mean(-log(0.9*1)) = -log(0.9) = 0.0458.

        :param predictions:  predictions of the target, made by the model.
        :param targets: true target.
        :return: the CCE loss, the average value of the log of the prediction made.
        """

        prediction_made_for_correct_target = np.sum(
            np.multiply(predictions, targets),
            axis=-1
        )

        return np.mean(-np.log(prediction_made_for_correct_target))

    def calculate_gradient(self, prediction, target):
        """
        Calculates the gradient of the loss with respect to the prediction. This is
        target/prediction_made_for_correct_class. For example, for prediction
        [0.1, 0.2, 0.5, 0.2] and target [0, 0, 1, 0], our gradient is
        [0, 0, 1, 0]/0.5 = [0, 0, 2, 0].

        :param prediction: prediction of the target, made by the model.
        :param target: true target.
        :return: the gradient of the CCE loss, with respect to the prediction made.
        """

        prediction = np.squeeze(prediction)
        target = np.squeeze(target)
        prediction_made_for_correct_target = np.dot(prediction, target)


        return -target/(prediction_made_for_correct_target + 1e-10)


class Mean_squared_error():
    """
    A class to calculate the MSE loss and the local gradients.
    """

    def calculate_loss(self, predictions, targets):
        """
        Calculates the loss. This is the average value of (target-prediction)^2.

        :param predictions:  predictions of the target, made by the model.
        :param targets: true target.
        :return: the MSE loss, the average value of (target-prediction)^2.
        """

        return np.mean((targets - predictions) ** 2)

    def calculate_gradient(self, predictions, targets):
        """
        Calculates the gradient of the loss function with respect to the prediction.

        :param predictions:  predictions of the target made by the model.
        :param targets: true target.
        :return: the gradient.
        """

        return -2 * np.mean(targets - predictions)