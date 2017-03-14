
import random
import sys
import numpy as np
from datetime import datetime
import gflags
from scipy import misc

from dataset import SegmentationDataSet, TrainingSetPreproc
from baseline_model import BaselineModel

gflags.DEFINE_boolean("notebook", False, "")

FLAGS = gflags.FLAGS

class Trainer:
  def __init__(self):
    self.S = BaselineModel.Settings()
    self.S.image_channels = 3
    self.S.batch_size = 10
    self.S.learning_rate = 1e-6
    self.S.learning_rate = 0.001
    self.S.num_conv_layers = 6
    self.S.num_dense_layers = 3
    self.S.image_width = 64 if FLAGS.notebook else 256
    self.S.image_height = 64 if FLAGS.notebook else 256

    self.ds = SegmentationDataSet()
    self.pp = TrainingSetPreproc(self.ds, self.S.image_width, self.S.image_height, self.S.image_channels)

    self.model = BaselineModel(self.S)

  def postprocess_batch(self, X, y):
    X = X.astype(np.float32) / 255.
    y = (y != 0).astype(np.uint8)
    return (X, y)

  def training_step(self, step):
    batch_images = np.random.randint(0, self.ds.get_training_set_size() - 1, self.S.batch_size)
    (X, y) = self.pp.make_training_batch(batch_images)
    (X, y) = self.postprocess_batch(X, y)

    (loss, accuracy) = self.model.fit(X, y)
    print("step %d: loss = %f, accuracy = %f" % (step, loss, accuracy))

  def validate(self, step):
    N = self.ds.get_validation_set_size()
    M = self.S.batch_size

    (X_val, y_val) = self.pp.make_validation_batch(np.arange(0, N))
    (X_val, y_val) = self.postprocess_batch(X_val, y_val)

    predict = np.zeros_like(y_val)

    for i in range(0, N // M + 1):
      l = i * M
      h = min((i + 1) * M, N)

      X = np.zeros((M, self.S.image_width, self.S.image_height, self.S.image_channels))
      X[0:h-l, :, :, :] = X_val[l:h, :, :, :]

      predict[l:h, :, :] = self.model.predict(X)[0:h-l, :, :]

    val_accuracy = np.mean(predict == y_val)

    tp = np.sum(np.logical_and(predict == 1, y_val == 1).astype(np.float32))
    fp = np.sum(np.logical_and(predict == 1, y_val == 0).astype(np.float32))
    fn = np.sum(np.logical_and(predict == 0, y_val == 1).astype(np.float32))
    tn = np.sum(np.logical_and(predict == 0, y_val == 0).astype(np.float32))

    P = tp / (tp + fp)
    R = tp / (tp + fn)

    F1 = 2. * P * R / (P + R)

    print("val_accuracy = %f, P = %f, R = %f, F1 = %f" % (val_accuracy, P, R, F1))
    print("tp = %d, fp = %d, tn = %d, fn = %d" % (tp, fp, tn, fn))

    image_to_save = (step // 10) % self.ds.get_validation_set_size()
    misc.imsave("debug/validation_%05d_image.png" % step, X_val[image_to_save, :, :])
    misc.imsave("debug/validation_%05d_mask.png" % step, y_val[image_to_save, :, :] * 200)
    misc.imsave("debug/validation_%05d_predict.png" % step, predict[image_to_save, :, :] * 200)

  def train(self, num_steps):
    start_time = datetime.now()

    for step in range(num_steps):
      self.training_step(step)

      if step % 10 == 0:
        self.validate(step)

        if step > 0:
          time_passed = float((datetime.now() - start_time).total_seconds())
          eta = num_steps * time_passed / step / 60 / 60

          print("eta = %.2f hours" % eta)

      sys.stdout.flush()

if __name__ == '__main__':
  FLAGS(sys.argv)
  Trainer().train(1000)

