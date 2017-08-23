
import os
import glob
import sqlite3
import numpy as np
import cv2

import common

from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

def _load_pts(file_path):
    with open(file_path, "rt") as f:
        lines = f.readlines()

    version = int(lines[0].split(": ")[1])
    assert version == 1

    n_points = int(lines[1].split(": ")[1])
    points = (x.split(" ") for x in lines[3:-1])
    points = [(float(x[0]), float(x[1])) for x in points]
    assert len(points) == n_points

    return np.array(points)

def _load_menpo_raw(train_dirs):
    for td in train_dirs:
        logger.info("Processing %s" % td)
        image_files = glob.glob(td + "/*.png") + glob.glob(td + "/*.jpg")
        for image_file in image_files:
            logger.info(image_file)
            pts = _load_pts(os.path.splitext(image_file)[0] + ".pts")
            image = cv2.imread(image_file)

def setup_menpo(cache_path, train_dirs):
    _load_menpo_raw(train_dirs)
    print("DONE")
