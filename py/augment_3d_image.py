from __future__ import division

import os
#import subprocess
#from subprocess import call
#import errno
#import itertools
#import sys
#import random
import numpy as np
import simpleITK as sitk

def main():

    nda = sitk.GetArrayFromImage(image)

    #rotating
    angleCount = 10
    angle = 360 / angleCount #The rotation angle in degrees.
    #axes : tuple of 2 ints, optional
    #The two axes that define the plane of rotation. Default is the first two axes.
    rotated = scipy.ndimage.interpolation.rotate(image, angle, axes)

    img = sitk.GetImageFromArray(nda)


if __name__ == "__main__":
    main()