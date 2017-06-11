
import sys
import unittest
import numpy as np
import tensorflow as tf
import gflags
from scipy import misc

FLAGS = gflags.FLAGS


class AutoEncoder:

    class Settings:
        dtype = tf.float32

        batch_size = 10
        image_width = 256
        image_height = 256
        image_channels = 1

        num_classes = 2

        num_conv_layers = 8
        num_conv_filters = 100

        kernel_width = 5
        kernel_height = 5

        learning_rate = 0.0001

    def __init__(self, settings):
        self.S = settings

        self.session = tf.Session()
        self.add_graph()

    def add_graph(self):
        self.X = tf.placeholder(self.S.dtype,
                                shape=[None,
                                       self.S.image_width,
                                       self.S.image_height,
                                       self.S.image_channels])

        Z = self.X
        for layer in range(self.S.num_conv_layers):
            Z = self.add_conv_layer("ConvLayer%d" % layer, Z,
                                    num_filters=self.S.num_conv_filters * (layer + 1))
            print(Z)

        for layer in reversed(range(self.S.num_conv_layers)):
            num_filters = self.S.num_conv_filters * layer
            activation = tf.nn.relu

            if layer == 0:
                num_filters = self.S.image_channels
                activation = None

            Z = self.add_deconv_layer(
                "DeconvLayer%d" % layer, Z, num_filters, activation)
            print(Z)

        self.decoded_image = Z

        WH = self.S.image_width * self.S.image_height

        print(self.X)
        print(Z)

        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.X - Z))
        print(self.loss)

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=self.S.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def add_conv_layer(self, name, inputs, num_filters):
        outputs = tf.layers.conv2d(
            name=name,
            inputs=inputs,
            filters=num_filters,
            kernel_size=[self.S.kernel_width, self.S.kernel_height],
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.zeros_initializer(),
            padding="SAME",
            strides=[2, 2],
            activation=tf.nn.relu)
        return outputs

    def add_deconv_layer(self, name, inputs, num_filters, activation=None):
        outputs = tf.layers.conv2d_transpose(
            name=name,
            inputs=inputs,
            filters=num_filters,
            kernel_size=[self.S.kernel_width, self.S.kernel_height],
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            bias_initializer=tf.zeros_initializer(),
            padding="SAME",
            strides=[2, 2],
            activation=activation)
        return outputs

    def fit(self, X):
        [_, loss] = self.session.run(
            [self.train_step, self.loss],
          feed_dict={self.X: X})

        return (loss)

    def predict(self, X):
        return self.decoded_image.eval(
            session=self.session, feed_dict={self.X: X})


class TestAutoEncoder(unittest.TestCase):

    def test_overfit(self):
        S = AutoEncoder.Settings()
        S.batch_size = 2
        model = AutoEncoder(S)

        X = np.random.randn(
            S.batch_size, S.image_width, S.image_height, S.image_channels)

        for i in range(1000):
            loss = model.fit(X)
            if i % 100 == 99 or i == 0:
                print("step %d: loss = %f" % (i, loss))

if __name__ == '__main__':
    FLAGS(sys.argv)
    unittest.main()
