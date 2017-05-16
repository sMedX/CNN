
import sys
import numpy as np
import gflags
import tensorflow as tf
import scipy.misc
from dataset import Dataset

FLAGS = gflags.FLAGS

gflags.DEFINE_integer("batch_size", 20, "")
gflags.DEFINE_integer("num_steps", 100, "")
gflags.DEFINE_float("keep_prob", 0.9, "")

class Model:
    num_input_channels = 3
    image_size = 224
    num_conv_filters = 10
    num_conv_layers = 5
    num_dense_units = 50
    kernel_size = 7
    learning_rate = 1e-5
    pos_loss_weight = 0.001

    num_brands = 10
    num_pos = 4

    def __init__(self):
        self.session = tf.Session()

        self.X = tf.placeholder(
            tf.float32, shape=[None, self.image_size, self.image_size, self.num_input_channels])

        self.y_brands = tf.placeholder(
            tf.float32, shape=[None, self.num_brands])
        self.y_pos = tf.placeholder(tf.float32, shape=[None, self.num_pos])

        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)

        print(self.X)
        print(self.y_brands)
        print(self.y_pos)

        Z = self.X
        for i in range(self.num_conv_layers):
            Z = self.conv2d("conv%d" %
                            i, Z, self.num_conv_filters * (i + 1), self.kernel_size)
            Z = self.max_pool_2x2(Z)
            print(Z)

        size = int(Z.shape[1])
        num_channels = int(Z.shape[3])
        Z = tf.reshape(Z, [-1, size * size * num_channels])
        print(Z)

        print("BRANDS:")
        self.output_brands = self.dropout(self.dense(Z, self.num_dense_units))
        self.output_brands = self.dense(self.output_brands, self.num_brands, activation=None)
        print(self.output_brands)
        self.loss_brands = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_brands, logits=self.output_brands))
        print(self.loss_brands)

        print("POS:")
        self.output_pos = self.dense(Z, self.num_pos, activation=None)
        print(self.output_pos)
        self.loss_pos = tf.reduce_sum(
            tf.square(self.output_pos - self.y_pos), axis=1)
        print(self.loss_pos)
        self.loss_pos = self.loss_pos * self.y_brands[:, 0]
        print(self.loss_pos)
        self.loss_pos = tf.reduce_mean(self.loss_pos)
        print(self.loss_pos)

        print("LOSS:")
        self.loss = self.loss_brands + self.loss_pos * self.pos_loss_weight
        print(self.loss)

        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

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

    def dense(self, X, num_units, activation=tf.nn.relu):
        return tf.layers.dense(
            inputs=X,
            units=num_units,
            activation=activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

    def fit(self, X, y_brands, y_pos):
        [_, loss_brands, loss_pos] = self.session.run(
            [self.train_step, self.loss_brands, self.loss_pos],
            feed_dict={self.X: X, self.y_brands: y_brands, self.y_pos: y_pos, self.is_training: True, self.keep_prob: FLAGS.keep_prob})
        return (loss_brands, loss_pos)

    def predict(self, X):
        [brands, pos] = self.session.run(
            [self.output_brands, self.output_pos],
            feed_dict={self.X: X, self.is_training: False, self.keep_prob: FLAGS.keep_prob})
        return (brands, pos)

def surface(l, t, r, b):
    return abs((r - l) * (b - t))

def train_model():
    train_ds = Dataset(list(range(1, 15)))
    validate_ds = Dataset(list([15, 16]))

    brands = train_ds.get_all_brands().union(validate_ds.get_all_brands())
    brands = sorted(list(brands))
    brands = dict(zip(brands, range(1, 1 + len(brands))))
    brands_inv = {v: k for k, v in brands.items()}
    brands_inv[0] = "NONE"
    print(brands)

    Model.num_brands = len(brands) + 1
    model = Model()

    for step in range(FLAGS.num_steps):
        X = np.zeros(
            (FLAGS.batch_size, Model.image_size, Model.image_size, Model.num_input_channels))
        y_brands = np.zeros((FLAGS.batch_size, len(brands) + 1))
        y_pos = np.zeros((FLAGS.batch_size, Model.num_pos))

        for i in range(FLAGS.batch_size):
            (image, label) = train_ds.get_random_image_and_label(
                Model.image_size)

            X[i, :, :, :] = image

            if label:
                (l, t, r, b, brand) = label
                y_brands[i, brands[brand]] = 1.
                y_pos[i, :] = np.array(
                    [l, t, r - l, b - l], dtype=np.float32) / Model.image_size
            else:
                y_brands[i, 0] = 1.
                y_pos[i, :] = -1

        if step % 10 == 0:
            labels = []
            for i in range(FLAGS.batch_size):
                (image, label) = validate_ds.get_random_image_and_label(
                    Model.image_size)
                X[i, :, :, :] = image
                labels.append(label)

            brand_pred, pos_pred = model.predict(X)

            brands_acc = 0
            pos_dice = 0
            for i in range(FLAGS.batch_size):
                label = labels[i]

                (l2, t2, r2, b2) = pos_pred[i]
                r2 += l2
                b2 += t2

                if label:
                    (l1, t1, r1, b1, brand) = label

                    l = max(l1, l2)
                    t = max(t1, t2)
                    r = min(r1, r2)
                    b = min(b1, b2)

                    pos_dice += 2. * surface(l, t, r, b) / (surface(l1, t1, r1, b1) + surface(l2, t2, r2, b2))
                else:
                    brand = "NONE"

                brands_acc += (brand == brands_inv[np.argmax(brand_pred[i])])

                if i == 0:
                    scipy.misc.imsave("debug/%06d_%s_%.2f_%.2f_%.2f_%.2f.jpg" %
                                      (step, brand, l2, t2, r2, b2), image)

            print("brands_acc = %f, pos_dice = %f" % (brands_acc / FLAGS.batch_size, pos_dice / FLAGS.batch_size))

        loss_brands, loss_pos = model.fit(X, y_brands, y_pos)

        print("step %d: loss_brands = %f, loss_pos = %f" %
              (step, loss_brands, loss_pos))

if __name__ == "__main__":
    FLAGS(sys.argv)
    train_model()
