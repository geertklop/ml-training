import silence_tensorflow.auto  # noqa: F401
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense


class MyModel(Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flat = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(num_classes, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flat(x)
        x = self.d1(x)
        return self.d2(x)


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape & Normalize data
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    # Create model instance
    model = MyModel(num_classes=10)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Fit
    model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

    # Evaluate
    model.evaluate(x_test, y_test, verbose=2)
