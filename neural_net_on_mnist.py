from Activations import Softmax
from Model import Model
from Layers import Dense
from Utils import train
from MNIST import MNIST_preprocessed

mnist = MNIST_preprocessed()
[[X_train, y_train], [X_test, y_test]] = mnist.fetch_data()

num_pixels = X_train.shape[1]
num_classes = y_train.shape[1]

model = Model(
    [
        Dense(num_pixels, input_shape=num_pixels),
        Dense(num_classes),
        Softmax()
    ]
)

model.compile_model("categorical_crossentropy", 0.01)

train(model=model,
      num_epochs=10,
      batch_size=1,
      dataset=[[X_train, y_train], [X_test, y_test]]
)