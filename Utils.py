import numpy as np

def train(model, num_epochs, batch_size, dataset):
    """
    A function used to train the neural network and output the performance at the end
    of each epoch. The dataset is shuffled every epoch.

    :param model: Model object from Model.py
    :param num_epochs: number of epochs the user wants the model to be trained on.
    :param batch_size: number of training examples the model will use to estimate
    the gradients, per update. Currently limited to one.
    :param dataset: dataset used for training. In the form [[X_train, y_train],
    [X_test, y_test]].
    """

    [features_train, target_train], [features_test, target_test] = dataset

    dataset_size = features_train.shape[0]
    num_batches_per_epoch = int(dataset_size / batch_size)

    for epoch in range(1, num_epochs + 1):
        p = np.random.permutation(dataset_size)
        features_train, target_train = features_train[p], target_train[p]

        for i in range(num_batches_per_epoch):
            features_train_batch = features_train[i * batch_size: (i + 1) * batch_size]
            target_train_batch = target_train[i * batch_size: (i + 1) * batch_size]

            # train model on the batch
            model.train_on_batch(features_train_batch, target_train_batch)


        # print the performance at the end of the epoch
        test_accuracy, test_loss = test_on_batch(model, features_test, target_test)
        train_accuracy, train_loss = test_on_batch(model, features_train, target_train)
        print("Epoch " + str(epoch) +
              ". Training loss: " + str(round(train_loss, 4)) +
              ", training accuracy: " + str(round(train_accuracy, 4)) +
              ", test loss: " + str(round(test_loss, 4)) +
              ", test accuracy: " + str(round(test_accuracy, 4)) + ".")

def test_on_batch(model, features, targets):
    """
    Function to give the performance on a given batch of data.

    :param model: Model object from Model.py.
    :param features: features to test the model on.
    :param targets: targets to test the model on.
    :return: loss and accuracy of the model, for that batch.
    """
    amount_examples = features.shape[0]
    accuracy = 0
    average_loss = 0

    for i in range(amount_examples):
        loss, correct = test_on_example(model, features[i], targets[i])
        average_loss += loss / amount_examples
        if correct is True:
            accuracy += 1/amount_examples

    return accuracy, average_loss

def test_on_example(model, x, target):
    """
    Function to test the model on one training example.

    :param model: Model object from Model.py.
    :param x: features to test the model on.
    :param target: target to test the model on.
    :return: the loss for that example and if the model's prediction was correct or not
    """
    NN_output, test_loss = model.forward_pass(x, target)

    if np.argmax(NN_output) == np.argmax(target):
        correct = True
    else:
        correct = False
    return test_loss, correct