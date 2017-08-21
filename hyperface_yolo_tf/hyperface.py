
import os

import unittest
import numpy as np
import tensorflow as tf
import time

IMAGE_SIZE = 224
IMAGE_CHANNELS = 3


class HyperFaceModel:

    def __init__(self, model_path=None):
        print("Loading model from", model_path)
        self.data_dict = np.load(model_path)

        self.X = tf.placeholder(tf.float32,
                                shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])

        self.conv1_1 = tf.nn.relu(self.conv_layer(self.X, "conv1_1"))
        self.conv1_2 = tf.nn.relu(self.conv_layer(self.conv1_1, "conv1_2"))
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = tf.nn.relu(self.conv_layer(self.pool1, "conv2_1"))
        self.conv2_2 = tf.nn.relu(self.conv_layer(self.conv2_1, "conv2_2"))
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = tf.nn.relu(self.conv_layer(self.pool2, "conv3_1"))
        self.conv3_2 = tf.nn.relu(self.conv_layer(self.conv3_1, "conv3_2"))
        self.conv3_3 = tf.nn.relu(self.conv_layer(self.conv3_2, "conv3_3"))
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = tf.nn.relu(self.conv_layer(self.pool3, "conv4_1"))
        self.conv4_2 = tf.nn.relu(self.conv_layer(self.conv4_1, "conv4_2"))
        self.conv4_3 = tf.nn.relu(self.conv_layer(self.conv4_2, "conv4_3"))
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = tf.nn.relu(self.conv_layer(self.pool4, "conv5_1"))
        self.conv5_2 = tf.nn.relu(self.conv_layer(self.conv5_1, "conv5_2"))
        self.conv5_3 = tf.nn.relu(self.conv_layer(self.conv5_2, "conv5_3"))
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.data_dict = None

    def avg_pool(self, inputs, name):
        return tf.nn.avg_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, inputs, name):
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, inputs, name):
        with tf.variable_scope(name):
            W = self.get_conv_weight(name)
            b = self.get_bias(name)

            Z = tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding='SAME') + b

            print(Z)
            return Z

    def fc_layer(self, inputs, name):
        with tf.variable_scope(name):
            W = self.get_fc_weight(name)
            b = self.get_bias(name)

            Z = tf.matmul(x, W) + b

            print(Z)
            return Z

    def get_conv_weight(self, name):
        W = self.data_dict[name + "/W"]
        W = np.swapaxes(W, 1, 2)
        W = np.swapaxes(W, 0, 3)
        return tf.constant(W, name="weight")

    def get_fc_weight(self, name):
        W = self.data_dict[name] + "/W"
        return tf.constant(W, name="weight")

    def get_bias(self, name):
        b = self.data_dict[name + "/b"]
        return tf.constant(b, name="bias")


class TestHyperFaceModel(unittest.TestCase):

    def test_load(self):
        model = HyperFaceModel("model.npz")


if __name__ == '__main__':
        unittest.main()
