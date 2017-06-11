#! /usr/bin/python3

import os
import sys
import re
from glob import glob
import random
import unittest
import numpy as np
import gflags
from scipy import misc

FLAGS = gflags.FLAGS

gflags.DEFINE_string("segmentation_dataset_dir", "/home/mel/datasets/ultrasound_mw/", "")
gflags.DEFINE_string("segmentation_dataset_image_glob", "features/*.png", "")
gflags.DEFINE_string("segmentation_dataset_mask_glob", "mask/*.png", "")
gflags.DEFINE_string("segmentation_dataset_validation_set_regex", ".*[2-3]_.*.png", "")

class DataSet:
  def get_training_set_size(self):
    raise NotImplementedError

  def get_training_pair(self, num):
    raise NotImplementedError

  def get_test_set_size(self):
    raise NotImplementedError

  def get_test_image(self, num):
    return NotImplementedError

class SegmentationDataSet(DataSet):
  def __init__(self):
    images = glob("%s/%s" % (FLAGS.segmentation_dataset_dir, FLAGS.segmentation_dataset_image_glob))
    images = {os.path.basename(x): x for x in images}

    masks = glob("%s/%s" % (FLAGS.segmentation_dataset_dir, FLAGS.segmentation_dataset_mask_glob))
    masks = {os.path.basename(x): x for x in masks}

    self.training_set = []
    self.validation_set = []
    self.test_set = []

    for key, image in images.items():
      if key in masks:
        if re.match(FLAGS.segmentation_dataset_validation_set_regex, key):
          self.validation_set.append(((image, masks[key])))
        else:
          self.training_set.append((image, masks[key]))
        del masks[key]
      else:
        self.test_set.append(image)

    assert len(masks) == 0, "Masks without an image: " + str(masks)

    print("Training set has %d images." % len(self.training_set))
    print(self.training_set)
    print("Validation set has %d images." % len(self.validation_set))
    print(self.validation_set)
    print("Test set has %d images." % len(self.test_set))
    print(self.test_set)

    self.cache = {}

  def get_pair(self, image_file, mask_file):
    if image_file in self.cache:
      return self.cache[image_file]

    image = misc.imread(image_file)
    if image.shape[2] == 4: image = image[:, :, 0:3]

    mask = misc.imread(mask_file)
    mask = np.logical_and(mask[:, :, 0] == 255, mask[:, :, 1] == 0, mask[:, :, 2] == 0)

    self.cache[image_file] = (image, mask)

    return (image, mask)

  def get_training_set_size(self):
    return len(self.training_set)

  def get_training_pair(self, num):
    (image_file, mask_file) = self.training_set[num]
    return self.get_pair(image_file, mask_file)

  def get_validation_set_size(self):
    return len(self.validation_set)

  def get_validation_pair(self, num):
    (image_file, mask_file) = self.validation_set[num]
    return self.get_pair(image_file, mask_file)

class TestSegmentationDataset(unittest.TestCase):
  def test_basic(self):
    ds = SegmentationDataSet()
    assert ds.get_training_set_size() > 0

    (image, mask) = ds.get_training_pair(0)
    (w1, h1, c1) = image.shape
    (w2, h2) = mask.shape
    assert w1 == w2 and h1 == h2, "Mismatching shapes: %dx%d vs. %dx%d." % (w1, h1, w2, h2)

class TrainingSetPreproc:
  def __init__(self, dataset, image_width, image_height, image_channels):
    self.dataset = dataset
    self.image_width = image_width
    self.image_height = image_height
    self.image_channels = image_channels

  def preprocess_pair(self, image, mask, norandom = False):
    (w1, h1, c1) = image.shape
    (w2, h2) = mask.shape
    assert w1 == w2 and h1 == h2, "Mismatching shapes: %dx%d vs. %dx%d." % (w1, h1, w2, h2)

    # resize if we need to
    k = max(float(self.image_width) / w1, float(self.image_height) / h1)
    if k != 1:
      size = (max(int(w1 * k), self.image_width), max(int(h1 * k), self.image_height))
      image = misc.imresize(image, size)
      mask = misc.imresize(mask, size, "nearest")
      (w1, h1, c1) = image.shape

    # crop
    if norandom:
      i = (w1 - self.image_width) // 2
      j = (h1 - self.image_height) // 2
    else:
      i = random.randint(0, w1 - self.image_width)
      j = random.randint(0, h1 - self.image_height)

    image = image[i : i + self.image_width, j : j + self.image_height]
    mask = mask[i : i + self.image_width, j : j + self.image_height]

    return (image, mask)

  def make_batch_with_getter(self, image_nums, getter, norandom = False):
    batch_size = image_nums.shape[0]

    X = np.zeros((batch_size, self.image_width, self.image_height, self.image_channels))
    y = np.zeros((batch_size, self.image_width, self.image_height))

    for i, num in enumerate(image_nums):
      (image, mask) = getter(num)
      (X[i, :, :, :], y[i, :, :]) = self.preprocess_pair(image, mask, norandom = norandom)

    return (X, y)

  def make_training_batch(self, image_nums):
    return self.make_batch_with_getter(image_nums, self.dataset.get_training_pair)

  def make_validation_batch(self, image_nums):
    return self.make_batch_with_getter(image_nums, self.dataset.get_validation_pair, norandom = True)

class TestTrainingSetPreparoc(unittest.TestCase):
  def test_basic(self):
    ds = SegmentationDataSet()
    pp = TrainingSetPreproc(ds, 128, 128, 3)
    (image, mask) = pp.make_training_batch(np.array([1]))
    (_, w1, h1, c1) = image.shape
    (_, w2, h2) = mask.shape
    assert np.max(np.unique(mask) - np.array([0, 255])) == 0, np.unique(mask)

if __name__ == "__main__":
  FLAGS(sys.argv)
  unittest.main()
