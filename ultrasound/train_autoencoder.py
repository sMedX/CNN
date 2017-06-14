
import os
import sys
import random
import gflags
import numpy as np
from glob import glob
from scipy import misc
from autoencoder import AutoEncoder

FLAGS = gflags.FLAGS
gflags.DEFINE_string("dataset_dir", "/home/mel/datasets/ultrasound_mw/", "")
gflags.DEFINE_string("debug_dir", "./debug", "")
gflags.DEFINE_string("checkpoints_dir", "./checkpoints", "")
gflags.DEFINE_integer("train_steps", 100, "")
gflags.DEFINE_integer("batch_size", 16, "")
gflags.DEFINE_integer("image_width", 256, "")
gflags.DEFINE_integer("image_height", 256, "")


class DataSet:

    def __init__(self):
        self.images = glob(os.path.join(FLAGS.dataset_dir, "*.jpeg"))
        print("Found %d images." % (len(self.images)))

    def get_random_image(self):
        image = random.choice(self.images)
        image = misc.imread(image)
        image = misc.imresize(image, (FLAGS.image_width, FLAGS.image_height))
        image = np.mean(image, axis=2, keepdims=True)
        return image

    def get_random_batch(self, size=None):
        if not size:
            size = FLAGS.batch_size
        X = np.zeros((size, FLAGS.image_width, FLAGS.image_height, 1))
        for i in range(size):
            X[i, :, :, :] = self.get_random_image()
        return X


class Trainer:

    def __init__(self):
        self.dataset = DataSet()

        self.S = AutoEncoder.Settings()
        self.S.image_width = FLAGS.image_width
        self.S.image_height = FLAGS.image_height
        self.S.image_channels = 1

        self.model = AutoEncoder(self.S)

    def train_step(self, step):
        batch = self.dataset.get_random_batch()
        loss = self.model.fit(batch, step)
        return loss

    def write_debug_image(self, step):
        X = self.dataset.get_random_batch(1)
        Y = self.model.predict(X)
        XY = np.concatenate((X, Y), axis = 1)
        XY = np.squeeze(XY, 0)
        XY = np.squeeze(XY, 2)
        misc.imsave(os.path.join(FLAGS.debug_dir, "%06d.jpeg" % step), XY)

    def train(self):
        for i in range(FLAGS.train_steps):
            loss = self.train_step(i)
            print("step %d, loss = %f, log_loss = %f" % (i, loss, np.log(loss)))
            if i % 20 == 19 or i == 0:
                self.write_debug_image(i)
            if i % 1000 == 999 or i == 0:
                self.model.write_model(os.path.join(FLAGS.checkpoints_dir, "%06d" % i))

def main():
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()
