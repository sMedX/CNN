#! /usr/bin/python3

import os
import sys
import gflags
import json
import unittest
import codecs
import random
import numpy as np
import scipy.misc

FLAGS = gflags.FLAGS

gflags.DEFINE_string("dataset_path", "/large/datasets/retail", "")
gflags.DEFINE_float("dataset_image_scale", 1.0, "")
gflags.DEFINE_float("dataset_pos_scale", 0.125, "")
gflags.DEFINE_boolean("dataset_norandom", False, "")
gflags.DEFINE_float("dataset_min_label_surface", 0.5, "")
gflags.DEFINE_string("crop_path", "/large/datasets/retail/crop", "")

class Collection:
    def __init__(self, basepath):
        self.basepath = basepath

        with open(os.path.join(basepath, "test.json"), "rb") as f:
            reader = codecs.getreader("utf-8")
            self.labels = json.load(reader(f))

        self.image_filenames = sorted(list(self.labels.keys()))

    def get_image_path(self, filename):
        return os.path.join(self.basepath, filename)

    def get_random_label(self):
        if FLAGS.dataset_norandom:
            filename = self.image_filenames[0]
        else:
            filename = random.choice(self.image_filenames)

        return (filename, self.labels[filename])

    def get_all_brands(self):
        brands = set()
        for _, labels in self.labels.items():
            for _, _, _, _, brand in labels:
                brands.add(brand)
        return brands

    def extract_label_pictures(self):
        for image_file_name, labels in self.labels.items():
            print("Processing %s" % image_file_name)
            image_file_path = os.path.join(self.basepath, image_file_name)
            image = scipy.misc.imread(image_file_path)

            for l, t, r, b, label in labels:
                if l < 0 or t < 0:
                    continue

                cropped_image = image[t:b, l:r]

                cropped_image_path = "%s_%s_%d_%d_%d_%d.jpg" % (
                    label, os.path.basename(image_file_name), l, t, r, b)
                cropped_image_path = os.path.join(
                    FLAGS.crop_path, cropped_image_path)

                print("Writing %s" % cropped_image_path)
                scipy.misc.imwrite(cropped_image_path, cropped_image)


class TestCollection(unittest.TestCase):
    def test_basic(self):
        col=Collection(os.path.join(FLAGS.dataset_path, "part1"))


class Dataset:
    def __init__(self, parts):
        print(parts)
        parts = [os.path.join(FLAGS.dataset_path, "part%d" % p) for p in parts]
        self.collections = [Collection(p) for p in parts]

    def get_random_patch_and_label(self, size):
        if FLAGS.dataset_norandom:
            collection = self.collections[0]
            [filename, labels] = collection.get_random_label()
        else:
            collection = random.choice(self.collections)
            [filename, labels] = collection.get_random_label()

        image = scipy.misc.imread(collection.get_image_path(filename))

        if FLAGS.dataset_image_scale != 1.0:
            image = scipy.misc.imresize(image, FLAGS.dataset_image_scale, 'bilinear')

        y = random.randint(0, image.shape[0] - size - 1)
        x = random.randint(0, image.shape[1] - size - 1)
        cropped_iamge = image[y : y+size, x : x+size]

        biggest_label = None
        biggest_label_size = FLAGS.dataset_min_label_surface * size * size

        for l, t, r, b, label in labels:
            l *= FLAGS.dataset_pos_scale
            t *= FLAGS.dataset_pos_scale
            r *= FLAGS.dataset_pos_scale
            b *= FLAGS.dataset_pos_scale

            l -= x
            t -= y
            r -= x
            b -= y

            if r < 0: continue
            if b < 0: continue
            if l > size: continue
            if t > size: continue

            if l < 0: l = 0
            if t < 0: t = 0
            if r > size: r = size
            if b > size: b = size

            label_size = (r - l) * (b - t)

            if label_size > biggest_label_size:
                biggest_label_size = label_size
                biggest_label = ((int(l), int(t), int(r), int(b), label))

        return (cropped_iamge, biggest_label)

    def get_random_patch_and_label_2(self, size):
        while True:
            collection = random.choice(self.collections)
            [filename, labels] = collection.get_random_label()
            image = scipy.misc.imread(collection.get_image_path(filename))
            image = scipy.misc.imresize(image, FLAGS.dataset_image_scale, 'bilinear')

            if random.randint(0, len(labels)) == 0:
                attempt = 0
                bad = True
                while attempt < 100 and bad:
                    y = random.randint(0, image.shape[0] - size - 1)
                    x = random.randint(0, image.shape[1] - size - 1)

                    attempt += 1

                    bad = False
                    for l, t, r, b, label in labels:
                        l *= FLAGS.dataset_pos_scale
                        t *= FLAGS.dataset_pos_scale
                        r *= FLAGS.dataset_pos_scale
                        b *= FLAGS.dataset_pos_scale

                        if r >= x and b >= y and l < x + size and t < y + size:
                            bad = True
                            break

                if bad: continue

                cropped_image = image[y:y+size, x:x+size]

                return (cropped_image, None)
            else:
                for attempt in range(100):
                    (l, t, r, b, label) = random.choice(labels)

                    l = max(0, int(FLAGS.dataset_pos_scale * l))
                    t = max(0, int(FLAGS.dataset_pos_scale * t))
                    r = min(image.shape[0] - 1, int(FLAGS.dataset_pos_scale * r))
                    b = min(image.shape[1] - 1, int(FLAGS.dataset_pos_scale * b))

                    if r - l <= size or b - t <= size: continue

                    cropped_image = image[t:b, l:r]

                    y = random.randint(0, cropped_image.shape[0] - size - 1)
                    x = random.randint(0, cropped_image.shape[1] - size - 1)
                    cropped_image = cropped_image[y:y+size, x:x+size]

                    return (cropped_image, label)

    def get_all_brands(self):
        brands = set()
        for col in self.collections:
            brands = brands.union(col.get_all_brands())
        return brands

    def get_random_image(self):
        collection = random.choice(self.collections)
        [filename, labels] = collection.get_random_label()
        image = scipy.misc.imread(collection.get_image_path(filename))
        return (image, labels)

class TestDataset(unittest.TestCase):
    def test_dataset(self):
        dataset = Dataset()

        for i in range(10):
            (image, label) = dataset.get_random_image_and_label(224)
            if label:
                scipy.misc.imsave("test_%d_%d_%d_%d_%s.jpeg" % label, image)
            else:
                scipy.misc.imsave("test_nolabel.jpeg", image)

if __name__ == "__main__":
    FLAGS(sys.argv)
    unittest.main()

    # for part in range(1, FLAGS.num_parts + 1):
    #     col = Collection(os.path.join(FLAGS.dataset_path, "part%d" % part))
    #     col.extract_label_pictures()
