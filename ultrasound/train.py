
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
FLAGS(sys.argv)


S = BaselineModel.Settings()
S.image_channels = 3
S.batch_size = 10
S.learning_rate = 1e-6
S.learning_rate = 0.001
S.num_conv_layers = 6
S.num_dense_layers = 3

if not FLAGS.notebook:
  S.image_width = 256
  S.image_height = 256
else:
  S.image_width = 64
  S.image_height = 64

ds = SegmentationDataSet()
pp = TrainingSetPreproc(ds, S.image_width, S.image_height, S.image_channels)

model = BaselineModel(S)

#num_steps = 50000
num_steps = 1000
start_time = datetime.now()

for step in range(num_steps):
  batch_images = np.random.randint(0, ds.get_training_set_size() - 1, S.batch_size)
  (X, y) = pp.make_training_batch(batch_images)
  X = X.astype(np.float32) / 255.
  y = (y != 0).astype(np.uint8)

  (loss, accuracy) = model.fit(X, y)

  print("step %d: loss = %f, accuracy = %f" % (step, loss, accuracy))

  if step % 10 == 0:
    (X_val, y_val) = pp.make_validation_batch(np.random.randint(0, ds.get_validation_set_size() - 1, S.batch_size))
    X_val = X_val.astype(np.float32) / 255.
    y_val = (y_val != 0).astype(np.uint8)

    predict = model.predict(X_val)
    val_accuracy = np.mean(predict == y_val)

    tp = np.sum(np.logical_and(predict == 1, y_val == 1).astype(np.float32))
    fp = np.sum(np.logical_and(predict == 1, y_val == 0).astype(np.float32))
    fn = np.sum(np.logical_and(predict == 0, y_val == 1).astype(np.float32))
    tn = np.sum(np.logical_and(predict == 0, y_val == 0).astype(np.float32))

    P = tp / (tp + fp)
    R = tp / (tp + fn)

    F2 = 2. * P * R / (P + R)

    eta = num_steps * float((datetime.now() - start_time).total_seconds()) / (step + 1) / 60 / 60

    print("val_accuracy = %f, P = %f, R = %f, F2 = %f, eta = %.2f hours" % (val_accuracy, P, R, F2, eta))
    print("tp = %d, fp = %d, tn = %d, fn = %d" % (tp, fp, tn, fn))

    image_to_save = random.randint(0, S.batch_size - 1)
    misc.imsave("debug/validation_%05d_image.png" % step, X_val[image_to_save, :, :])
    misc.imsave("debug/validation_%05d_mask.png" % step, y_val[image_to_save, :, :] * 200)
    misc.imsave("debug/validation_%05d_predict.png" % step, predict[image_to_save, :, :] * 200)

  sys.stdout.flush()
