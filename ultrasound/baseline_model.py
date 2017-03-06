
import sys
import unittest
import numpy as np
import tensorflow as tf
import gflags
from scipy import misc

FLAGS = gflags.FLAGS

class BaselineModel:
  class Settings:
    dtype = tf.float32

    batch_size = 10
    image_width = 256
    image_height = 256
    image_channels = 1

    num_classes = 2

    num_conv_layers = 2
    num_conv_filters = 100

    kernel_width = 5
    kernel_height = 5

    num_dense_layers = 2
    num_dense_filters = 40

    learning_rate = 1e-3

  def __init__(self, settings):
    self.S = settings

    self.session = tf.Session()
    self.add_graph()

  def add_graph(self):
    self.X = tf.placeholder(self.S.dtype,
                            shape = (self.S.batch_size,
                                     self.S.image_width,
                                     self.S.image_height,
                                     self.S.image_channels))
    self.y = tf.placeholder(tf.uint8,
                            shape = (self.S.batch_size,
                                     self.S.image_width,
                                     self.S.image_height,
                                     1))

    self.layer_activations = []

    activations = self.add_conv_layer("ConvLayer0", self.X)
    print(activations)
    self.layer_activations.append(activations)

    for layer in range(1, self.S.num_conv_layers):
      activations = self.add_conv_layer("ConvLayer%d" % layer, activations)
      activations = tf.contrib.layers.max_pool2d(activations, 2)
      print("conv", activations)
      self.layer_activations.append(activations)

    for layer in range(1, self.S.num_conv_layers):
      layer_activations = self.add_deconv_layer("DeconvLayer%d" % layer, activations)
      prev_activations = self.layer_activations[self.S.num_conv_layers - 1 - layer]
      print(layer_activations)

      activations = tf.concat([layer_activations, prev_activations], axis = 3)
      print("deconv", activations)

      self.layer_activations.append(activations)


    for layer in range(0, self.S.num_dense_layers):
      last = (layer == self.S.num_dense_layers - 1)

      activations = self.add_dense_layer("DenseLayer%d" % layer,
                                         activations,
                                         self.S.num_classes if last else self.S.num_dense_filters,
                                         None if last else tf.nn.relu)
      print(activations)

    WH = self.S.image_width * self.S.image_height

    activations = tf.reshape(activations, [self.S.batch_size * WH, -1])
    print(activations)

    self.prediction = tf.cast(tf.argmax(activations, axis = 1), tf.uint8)
    self.prediction = tf.reshape(self.prediction, [self.S.batch_size, self.S.image_width, self.S.image_height])

    self.y_one_hot_flat = tf.one_hot(tf.reshape(self.y, [-1]), self.S.num_classes)
    print(self.y_one_hot_flat)

    self.loss = tf.nn.softmax_cross_entropy_with_logits(
      labels = self.y_one_hot_flat,
      logits = activations)
    self.loss = tf.reduce_mean(self.loss)
    print(self.loss)

    self.train_step = tf.train.AdamOptimizer(
      learning_rate = self.S.learning_rate).minimize(self.loss)

    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y, [-1]), tf.reshape(self.prediction, [-1])), tf.float32))
    print(self.accuracy)

    self.session.run(tf.global_variables_initializer())

  def add_conv_layer(self, name, inputs):
    print(name, inputs)
    outputs = tf.layers.conv2d(
      name = name,
      inputs = inputs,
      filters = self.S.num_conv_filters,
      kernel_size = [self.S.kernel_width, self.S.kernel_height],
      kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
      bias_initializer = tf.zeros_initializer(),
      padding = "SAME",
      activation = tf.nn.relu)
    return outputs

  def add_deconv_layer(self, name, inputs):
    print(name, inputs)
    outputs = tf.layers.conv2d_transpose(
      name = name,
      inputs = inputs,
      filters = self.S.num_conv_filters,
      kernel_size = [self.S.kernel_width, self.S.kernel_height],
      kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
      bias_initializer = tf.zeros_initializer(),
      padding = "SAME",
      strides = [2, 2],
      activation = tf.nn.relu)

    return outputs

  def add_dense_layer(self, name, inputs, num_outputs, activation):
    print(name, inputs, num_outputs, activation)
    outputs = tf.layers.conv2d(
      name = name,
      inputs = inputs,
      filters = num_outputs,
      kernel_size = [1, 1],
      kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
      bias_initializer = tf.zeros_initializer(),
      padding = "SAME",
      activation = activation)

    return outputs

  def fit(self, X, y):
    y = np.expand_dims(y, 3)

    [_, loss, accuracy] = self.session.run(
      [self.train_step, self.loss, self.accuracy],
      feed_dict = { self.X: X, self.y: y })

    return (loss, accuracy)

  def predict(self, X):
    prediction = self.prediction.eval(session = self.session, feed_dict = { self.X: X })
    return prediction

class TestBaselineModel(unittest.TestCase):
  def test_overfit(self):
    S = BaselineModel.Settings()
    S.batch_size = 2
    model = BaselineModel(S)

    X = np.random.randn(S.batch_size, S.image_width, S.image_height, S.image_channels)
    y = np.sign(np.random.rand(S.batch_size, S.image_width, S.image_height)) * 2. + 1.

    X[:, :, :, 0] -= .1 * y

    for i in range(5):
      (loss, accuracy) = model.fit(X, y)
      print("step %d: loss = %f, accuracy = %f" % (i, loss, accuracy))

if __name__ == '__main__':
  FLAGS(sys.argv)
  unittest.main()
