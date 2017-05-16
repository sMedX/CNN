#! /usr/bin/python3

import sys
import numpy as np
import gflags
import tensorflow as tf
import scipy.misc
from dataset import Dataset
from segmenter import Segmenter

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("batch_size", 20, "")
gflags.DEFINE_integer("num_steps", 100, "")
gflags.DEFINE_float("dropout_keep_prob", 0.9, "")
gflags.DEFINE_boolean("batch_norm", False, "")

class Model:
    class Settings:
        num_input_channels = 3
        num_conv_filters = 20
        num_conv_layers = 4
        num_dense_layers = 2
        num_dense_units = 50
        kernel_size = 3
        learning_rate = 1e-5
        num_brands = 10
        class_weights = [1] + 9 * [2]
        image_size = 32
        keep_prob = 0.9

    def __init__(self, settings):
        self.S = settings

        self.session = tf.Session()

        self.X = tf.placeholder(
            tf.float32, shape=[None, self.S.image_size, self.S.image_size, self.S.num_input_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, self.S.num_brands])

        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        print(self.X)
        print(self.y)

        Z = self.X
        print(Z)
        for i in range(self.S.num_conv_layers):
            Z = self.batch_norm(Z)
            print(Z)

            Z = self.conv2d("conv%d_a" % i, Z,
                            self.S.num_conv_filters * (i + 1),
                            self.S.kernel_size)
            print(Z)

            Z = self.conv2d("conv%d_b" % i, Z,
                            self.S.num_conv_filters * (i + 1),
                            self.S.kernel_size)
            print(Z)

            Z = self.max_pool_2x2(Z)
            print(Z)

        size = int(Z.shape[1])
        num_channels = int(Z.shape[3])
        Z = tf.reshape(Z, [-1, size * size * num_channels])
        print(Z)

        Z = self.dropout(Z)
        print(Z)

        Z = self.dense(Z, self.S.num_dense_units)
        print(Z)

        self.output = self.dense(Z, self.S.num_brands, activation=None)
        print(self.output)

        y_num = tf.argmax(self.y, axis = 1)
        output_num = tf.argmax(self.output, axis = 1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(y_num, output_num), tf.float32))
        print("accuracy =", self.accuracy)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.output)
        class_weights = tf.constant(np.array(self.S.class_weights, dtype=np.float32))
        loss_weight = tf.multiply(self.y, class_weights)
        loss_weight = tf.reduce_sum(loss_weight, axis=1)
        self.loss = tf.reduce_mean(tf.multiply(loss, loss_weight))
        print("loss =", self.loss)

        self.train_step = tf.train.AdamOptimizer(self.S.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, name, X, num_filters, kernel_size):
        return tf.layers.conv2d(
            name=name,
            inputs=X,
            filters=num_filters,
            kernel_size=[kernel_size, kernel_size],
            activation=tf.nn.relu,
            padding='SAME',
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    def max_pool_2x2(self, X):
        return tf.nn.max_pool(X, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def dropout(self, inputs):
        return tf.cond(
            self.is_training,
            lambda: tf.nn.dropout(inputs, self.keep_prob),
            lambda: inputs)

    def batch_norm(self, inputs):
        if FLAGS.batch_norm:
            return tf.layers.batch_normalization(inputs, training=self.is_training)
        else:
            return inputs

    def dense(self, X, num_units, activation=tf.nn.relu):
        return tf.layers.dense(
            inputs=X,
            units=num_units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

    def fit(self, X, y):
        [_, loss, accuracy] = self.session.run(
            [self.train_step, self.loss, self.accuracy],
            feed_dict={self.X: X, self.y: y, self.is_training: True, self.keep_prob: self.S.keep_prob})
        return loss, accuracy

    def predict(self, X):
        [brands] = self.session.run(
            [self.output],
            feed_dict={self.X: X, self.is_training: False, self.keep_prob: self.S.keep_prob})
        return brands

class Trainer:
    def __init__(self):
        self.train_ds = Dataset(list(range(1, 15)))
        self.validate_ds = Dataset(list([15, 16]))

        self.brands = self.train_ds.get_all_brands().union(self.validate_ds.get_all_brands())
        self.brands = sorted(list(self.brands))
        self.brands = dict(zip(self.brands, range(1, 1 + len(self.brands))))
        self.brands_inv = {v: k for k, v in self.brands.items()}
        self.brands_inv[0] = "NONE"
        print(self.brands)
        print(self.brands_inv)

        self.settings = Model.Settings()
        self.settings.num_brands = len(self.brands) + 1
        self.settings.keep_prob = FLAGS.dropout_keep_prob
        #self.settings.class_weights = [1] + [3]*len(self.brands)
        #self.settings.class_weights = [1] * (1+len(self.brands))
        self.settings.class_weights = [3] + [1]*len(self.brands)

        self.model = Model(self.settings)

    def fill_training_batch(self, X, y, step):
        S = self.settings

        for i in range(FLAGS.batch_size):
            # (image, label) = self.train_ds.get_random_patch_and_label(S.image_size)

            # X[i, :, :, :] = image

            # if label:
            #     (l, t, r, b, brand) = label
            #     y[i, self.brands[brand]] = 1.
            # else:
            #     y[i, 0] = 1.

            (image, brand) = self.train_ds.get_random_patch_and_label_2(S.image_size)
            X[i, :, :, :] = image
            if brand: y[i, self.brands[brand]] = 1.
            else: y[i, 0] = 1.

            if i == 0:
                scipy.misc.imsave("debug/train_%06d_%s.jpg" % (step, brand),
                                  X[0, :, :, :])


    def validate_one_batch(self, X, step):
        S = self.settings

        labels = []
        for i in range(FLAGS.batch_size):
            # (image, label) = self.validate_ds.get_random_patch_and_label(
            #     S.image_size)
            # X[i, :, :, :] = image
            # labels.append(label)

            (image, brand) = self.train_ds.get_random_patch_and_label_2(S.image_size)
            X[i, :, :, :] = image
            labels.append(brand)

            if i == 0:
                scipy.misc.imsave("debug/valid_%06d_%s.jpg" % (step, brand),
                                  X[0, :, :, :])

        pred = self.model.predict(X)

        brands_acc = 0
        for i in range(FLAGS.batch_size):
            # label = labels[i]
            # if label:
            #     (l1, t1, r1, b1, brand_actual) = label
            # else:
            #     brand_actual = "NONE"

            brand_actual = labels[i]
            if not brand_actual: brand_actual = "NONE"

            brand_pred = self.brands_inv[np.argmax(pred[i, :])]
            brands_acc += (brand_actual == brand_pred)

            # if i == 0:
            #     if brand_actual != "NONE":
            #         scipy.misc.imsave("debug/%06d_%s_%d_%d_%d_%d_%s.jpg" %
            #                           (step, brand_actual, l1, t1, r1, b1, brand_pred),
            #                           image)
            #     else:
            #         scipy.misc.imsave("debug/%06d_%s_%s.jpg" %
            #                           (step, brand_actual, brand_pred), image)

        print("validation set accuracy = %f" % (brands_acc / FLAGS.batch_size))

    def train(self):
        S = self.settings

        for step in range(FLAGS.num_steps):
            X = np.zeros((FLAGS.batch_size, S.image_size, S.image_size,
                          S.num_input_channels))
            y = np.zeros((FLAGS.batch_size, len(self.brands) + 1))
            self.fill_training_batch(X, y, step)

            loss, accuracy = self.model.fit(X, y)
            print("step %d: loss = %f, accuracy = %f" % (step, loss, accuracy))

            if step % 10 == 0:
                self.validate_one_batch(X, step)

                segmenter = Segmenter(self.model, S.image_size, S.num_brands)
                (image, labels) = self.validate_ds.get_random_image()
                image = segmenter.segment_and_color_image(image)
                scipy.misc.imsave("debug/%06d_colored.jpg" % (step), image)

if __name__ == "__main__":
    FLAGS(sys.argv)
    Trainer().train()
