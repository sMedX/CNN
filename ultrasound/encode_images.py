
import os
import sys
import gflags
import numpy as np
import pickle
from glob import glob
from scipy import misc
from autoencoder import AutoEncoder
from train_autoencoder import DataSet

FLAGS = gflags.FLAGS
gflags.DEFINE_string("model", "", "")
gflags.DEFINE_string("output", "", "")

FLAGS(sys.argv)

settings = AutoEncoder.Settings()
settings.image_width = FLAGS.image_width
settings.image_height = FLAGS.image_height
settings.image_channels = 1

model = AutoEncoder(settings)
model.read_model(FLAGS.model)

output = {}

dataset = DataSet()
for index, filename in enumerate(dataset.images):
    image = dataset.read_image(filename)
    image = np.expand_dims(image, 0)

    Y = model.encode(image)
    Y = Y[0, 0, 0]

    print("%10d\t%3.2f%%\t%s" % (index, float(index)/len(dataset.images) * 100, filename))

    output[filename] = Y

pickle.dump(output, open(FLAGS.output, "wb"))

