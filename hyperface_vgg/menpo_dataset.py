
import os
import glob
import sqlite3
import numpy as np
import cv2
import chainer

import common

from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

IMG_SIZE = (227, 227)


def _load_pts(file_path):
    with open(file_path, 'rt') as f:
        lines = f.readlines()

    version = int(lines[0].split(': ')[1])
    assert version == 1

    n_points = int(lines[1].split(': ')[1])
    points = (x.split(' ') for x in lines[3:-1])
    points = [(float(x[0]), float(x[1])) for x in points]
    assert len(points) == n_points

    return np.array(points)


def _load_menpo_raw(train_dirs):
    image_files = []
    image_points = []
    for td in train_dirs:
        logger.info('Processing "{}"'.format(td))
        for image_file in glob.glob(td + '/*.png') + glob.glob(td + '/*.jpg'):
            logger.info(image_file)

            pts = _load_pts(os.path.splitext(image_file)[0] + '.pts')

            height = np.amax(pts[:, 0]) - np.amin(pts[:, 0])
            width = np.amax(pts[:, 1]) - np.amin(pts[:, 1])
            size = width * height

            if size < 50*50 or width < 50 or height < 50:
                logger.info('Skipping image (too small)')
                continue

            image_files.append(image_file)
            image_points.append(pts)

    image_files = np.array(image_files)
    image_points = np.array(image_points)

    return image_files, image_points


class MENPO(chainer.dataset.DatasetMixin):

    def __init__(self, image_files, image_points):
        self.image_files = image_files
        self.image_points = image_points

    def get_example(self, i):
        image_file = self.image_files[i]
        image = cv2.imread(image_file)
        image = image.astype(np.float32)
        image = cv2.normalize(image, None, -0.5, 0.5, cv2.NORM_MINMAX)

        points = self.image_points[i].astype(np.float32)
        min_x = np.amin(points[:, 0])
        max_x = np.amax(points[:, 0])
        min_y = np.amin(points[:, 1])
        max_y = np.amax(points[:, 1])

        size = max(max_x - min_x, max_y - min_y)
        max_y = min_y + size
        max_x = min_x + size

        image = image[int(min_y):int(max_y), int(min_x):int(max_x), :].copy()
        points[:, 0] = (points[:, 0] - min_x) / size
        points[:, 1] = (points[:, 1] - min_y) / size

        image = cv2.resize(image, IMG_SIZE)
        image = np.transpose(image, (2, 0, 1))

        if points.shape[0] == 39:
            points = np.concatenate((points,
                                     np.zeros((68 - 39, 2), dtype = np.float32)))
            mask = np.concatenate((np.ones((39, 2), dtype = np.float32),
                                   np.zeros((68 - 39, 2), dtype = np.float32)))
        elif points.shape[0] == 68:
            mask = np.ones((68, 2), dtype = np.float32)
        else:
            raise Exception(
                'Image "{}" has unknown landmakrs with {} points'.format(image_file, points.shape))

        points = points.reshape(-1)
        mask = mask.reshape(-1)

        return {'x_img': image,
                't_menpo_landmark': points,
                'm_menpo_landmark': mask}

    def __len__(self):
        return self.image_files.shape[0]


def setup_menpo(cache_path, train_dirs, test_rate):
    logger.info('Try to load MENPO cache from "{}"'.format(cache_path))
    try:
        cache_data = np.load(cache_path)
        image_files = cache_data['image_files']
        image_points = cache_data['image_points']
        order = cache_data['order']
        n_train = int(cache_data['n_train'])
    except (FileNotFoundError, KeyError):
        image_files, image_points = _load_menpo_raw(train_dirs)
        order = np.random.permutation(image_files.shape[0])

        n_train = int(image_files.shape[0] * (1.0 - test_rate))

        logger.info('Save MENPO cache to "{}"'.format(cache_path))
        np.savez(cache_path, image_files=image_files,
                 image_points=image_points,
                 order=order,
                 n_train=n_train)

    menpo = MENPO(image_files, image_points)

    train, test = chainer.datasets.split_dataset(menpo, n_train, order=order)
    logger.info('MENPO datasets (n_train:{}, n_test:{})'.
                format(len(train), len(test)))

    return train, test
