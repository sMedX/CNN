from __future__ import division

import random
import caffe
import math

import sys
import SimpleITK as sitk
import numpy as np
import os
import os.path
import scipy.misc
import scipy.ndimage
import scipy.ndimage.interpolation
import lmdb

f = open(listFile, 'r')
inputDirs = [i.replace('\n', '') for i in f.readlines()]
f.close()

print inputDirs

net = caffe.Classifier(path_to_prototxt,path_to_model)
net.set_phase_test()
net.set_mode_gpu()
scores = net.predict(img in form [1,x,y,1] or [1,x,y,3])