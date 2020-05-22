import silence_tensorflow.auto
import tensorflow as tf


class CustomDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight("weight", shape=[d, self.units])
        self.b = self.add_weight("bias", shape=[1, self.units])

    def call(self, x):
        # Forward linear pass
        z = tf.add(tf.matmul(x, self.W), self.b)

        # Activation
        y = tf.sigmoid(z)

        return y


if __name__ == "__main__":
    tf.random.set_seed(1)

    layer = CustomDenseLayer(3)
    layer.build((1, 2))

    x_input = tf.constant([[1, 2.]], shape=(1, 2))
    y = layer.call(x_input)

    print(y.numpy())
