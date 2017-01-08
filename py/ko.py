from __future__ import division

import math
import os
import os.path
import random
import sys

import SimpleITK as sitk
import caffe
import lmdb
import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.ndimage.interpolation


class AugmentLayer(caffe.Layer):
    """A layer that just performs ax+b, where a and b are
    random from [a_min, a_max], [b_min, b_max] respectively"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        self.sz = 128
        top[0].reshape(bottom[0].data.shape[0], 1, self.sz, self.sz)

    def forward(self, bottom, top):
        img = bottom[0].data
        # rotate
        degree = random.uniform(0, 179)
        img = scipy.ndimage.interpolation.rotate(img, degree, (2, 3))

        # crop & resize
        sz = (int(128 * random.triangular(0.7, 1.3)) / 2) * 2
        pad = (256 - sz) / 2
        img = img[:, :, pad:sz + pad, pad:sz + pad]  # todo sz

        img2 = np.zeros((bottom[0].data.shape[0], 1, self.sz, self.sz))

        for i in range(bottom[0].data.shape[0]):
            img2[i, 0, :, :] = scipy.misc.imresize(img[i, 0, :, :], (self.sz, self.sz))

        img = img2

        # contast
        a = random.triangular(0.5, 1.5)
        b = random.triangular(-75, 75)

        img = a * img + b

        # add noise
        img += 0.3 * np.std(top[0].data) * np.random.rand((bottom[0].data.shape[0], 1, self.sz, self.sz))

        top[0].data[...] = img

    def backward(self, top, propagate_down, bottom):
        pass


class CropLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[1].data.shape)

    def forward(self, bottom, top):
        sizeOld = bottom[0].data.shape[4]
        sizeNew = bottom[1].data.shape[4]
        offset = (sizeOld - sizeNew) / 2
        top[0].data[...] = bottom[0].data[:, :, offset + sizeNew, offset + sizeNew]

    def backward(self, top, propagate_down, bottom):
        pass


class Data3DCutter(caffe.Layer):
    """ cuts image by tiles
    creates 2.5d images as rgb images
    or 2d image as greyscaled images"""

    def setup(self, bottom, top):
        # preprocessExe = os.environ['preproc'] todo ?

        imageName, labelName, maskName, listFile = '', '', '', ''
        radius, stride, preset, strideNegative, spacingXY = None, None, '', None, 0.0
        try:
            imageName = 'patient.nrrd_preproc.nrrd'  # sys.argv[sys.argv.index('--imageName') + 1]
            labelName = 'livertumors.nrrd_preproc.nrrd'  # sys.argv[sys.argv.index('--labelName1') + 1]
            maskName = 'liver.nrrd_preproc.nrrd'  # sys.argv[sys.argv.index('--maskName') + 1]
            # listFile ='/root/host/caffe_nets/livertumors/a/train-cv-1-2.txt'#sys.argv[sys.argv.index('--listFile') + 1]
            listFile = '/root/host/train-cv-1-2.txt'  # sys.argv[sys.argv.index('--listFile') + 1]

            self.radius = 32  # int(sys.argv[sys.argv.index('--radius') + 1])
            stride = 3  # int(sys.argv[sys.argv.index('--stride') + 1].split(' ')[0])  # todo
            # preset = 'livertumors'#sys.argv[sys.argv.index('--preset') + 1]
            self.strideNegative = 6  # int(sys.argv[sys.argv.index('--strideNegative') + 1])
            # spacingXY = 0.8#float(sys.argv[sys.argv.index('--spacingXY') + 1])
        except Exception as e:
            print e
            exit(1)

        f = open(listFile, 'r')
        inputDirs = [i.replace('\n', '') for i in f.readlines()]
        f.close()

        print inputDirs

        imageDir = os.path.dirname(listFile)
        # check files
        inputPathes = []
        for inputDir in inputDirs:
            imagePath = os.path.join(imageDir, inputDir, imageName)
            if not os.path.isfile(imagePath):
                print 'no file: ', imagePath
                exit(2)
            labelPath = os.path.join(imageDir, inputDir, labelName)
            if not os.path.isfile(labelPath):
                print 'no file: ', labelPath
                exit(2)
            maskPath = os.path.join(imageDir, inputDir, maskName)
            if not os.path.isfile(maskPath):
                print 'no file: ', maskPath
                exit(2)
            inputPathes.append([imagePath, labelPath, maskPath])

        self.data = []
        for imagePath, labelPath, maskPath in inputPathes:
            n = imagePath.split(os.path.sep)[-2]

            image = sitk.ReadImage(imagePath, sitk.sitkUInt8)
            label = sitk.ReadImage(labelPath, sitk.sitkUInt8)
            mask = sitk.ReadImage(maskPath, sitk.sitkUInt8)

            # zyx ordered
            image = sitk.GetArrayFromImage(image)  # todo explicit uint8 type
            label = sitk.GetArrayFromImage(label)
            mask = sitk.GetArrayFromImage(mask)

            sz = np.array(image.shape)

            indices = np.array(mask.nonzero())[::, ::stride]

            # find bounding box
            l = np.amin(indices, axis=1)
            u = np.amax(indices, axis=1)

            # crop around bounding+r
            r = self.radius
            cropL = l - r
            cropL[cropL < 0] = 0

            cropU = u + r
            cropU[cropU >= sz] = sz[cropU >= sz] - 1

            image = image[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
            label = label[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
            mask = mask[cropL[0]:cropU[0], cropL[1]:cropU[1], cropL[2]:cropU[2]]
            print 'new shape ', image.shape

            # pad
            # wtf with +1?
            padL = r - l + 1
            padL[padL < 0] = 0

            padU = u + r - sz + 1
            padU[padU < 0] = 0

            pad = ((padL[0], padU[0]), (padL[1], padU[1]), (padL[2], padU[2]))

            image = np.pad(image, pad, mode='constant')
            label = np.pad(label, pad, mode='constant')
            mask = np.pad(mask, pad, mode='constant')

            print 'new shape ', image.shape
            # indiciesPos = np.array(label.nonzero()) todo
            indicesAll = np.array(mask.nonzero())[:, ::stride]
            indicesAll = list(indicesAll.T)

            print 'index count ', len(indicesAll)

            self.data.append([image, label, indicesAll])

    def reshape(self, bottom, top):
        self.batchSize = 32  # todo
        top[0].reshape(self.batchSize, 1, 128, 128)  # todo2*self.radius, 2*self.radius)
        top[1].reshape(self.batchSize, 1, 1, 1)

    def forward(self, bottom, top):
        negCount = 0
        for i in range(0, self.batchSize):
            image, label, indicesAll = random.choice(self.data)

            index = random.choice(indicesAll)
            iLabel = label[index[0], index[1], index[2]]

            if iLabel == 0:
                negCount += 1
                if negCount % self.strideNegative != 0:
                    while True:
                        index = random.choice(indicesAll)
                        iLabel = label[index[0], index[1], index[2]]
                        if iLabel == 1:
                            break

            print index, ":", iLabel
            r = 32  # todo
            tile = image[index[0], index[1] - r:index[1] + r, index[2] - r:index[2] + r]
            tile = scipy.ndimage.zoom(tile, 2, order=1)

            top[0].data[i, 0, :, :] = tile
            top[1].data[i, 0, 0, 0] = iLabel

    def backward(self, top, propagate_down, bottom):
        pass


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        self.diff[...] = bottom[0].data - bottom[1].data
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


class CrossEntropyLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        def sigmoid(x):
            try:
                return 1. / (1. + math.exp(-x))
            except:
                print 'overflow', x

        self.sigmoid = np.vectorize(sigmoid)
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs")
        self.weight = [1, 1]

    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].count != bottom[1].count:
        # raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        # self.diff[...] = bottom[0].data - bottom[1].data
        # top[0].data[...] = -np.sum(self.diff**2) / bottom[0].num
        # count = bottom[0].count
        num = bottom[0].num
        our = bottom[0].data
        target = bottom[1].data
        loss = 0
        for i in range(num):  # index of sample in batch
            for j in range(0, bottom[0].shape[1]):  # index of class
                our_ = self.sigmoid(our[i, j])
                try:
                    loss -= self.weight[j] * (target[i] * math.log(our_) + (1 - target[i]) * math.log(1 - our_))
                except:
                    print '*our_', our_
                    print '*our', our[i, j]
        top[0].data[0] = loss / num

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception(" Layer cannot backpropagate to label inputs.")

        if propagate_down[0]:
            # count = bottom[0].count
            num = bottom[0].num
            our = bottom[0].data
            target = bottom[1].data
            for i in range(num):  # *bottom[0].shape[2]*bottom[0].shape[3]
                label = int(target[i])
                target_vec = np.zeros(2)  # bottom[0].shape[1]
                target_vec[label] = 1
                our_ = self.sigmoid(our[i, :])
                print 'target', target_vec
                print 'our', our_
                bottom[0].diff[i, ...] = self.weight[label] * (our_ - target_vec) * top[0].diff[0] / num
                print 'diff', bottom[0].diff[i, :]

                # self.diff[...] = bottom[0].data - bottom[1].data


class LabelVecLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        class_count = 2  # todo
        top[0].reshape(bottom[0].shape[0], class_count)

    def forward(self, bottom, top):
        for i in range(bottom[0].shape[0]):
            label = int(bottom[0].data[i])
            target_vec = np.zeros(2)
            target_vec[label] = 1
            top[0].data[i, :] = target_vec

    def backward(self, top, propagate_down, bottom):
        pass


class LabelVecSpatialLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        class_count = 2  # todo
        top[0].reshape(bottom[0].shape[0], class_count, bottom[0].shape[2], bottom[0].shape[3])

    def forward(self, bottom, top):
        for i in range(bottom[0].shape[0]):  # *bottom[0].shape[2]*bottom[0].shape[3]
            for j in range(bottom[0].shape[2]):  # in w
                for k in range(bottom[0].shape[3]):  # in h
                    label = int(bottom[0].data[i, 0, j, k])
                    target_vec = np.zeros(2)  # bottom[0].shape[1]
                    target_vec[label] = 1
                    top[0].data[i, :, j, k] = target_vec

    def backward(self, top, propagate_down, bottom):
        pass


class ClassBalancerLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.weight = [1.0 / 99.7, 1.0 / 0.03]
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data

    def backward(self, top, propagate_down, bottom):
        for i in range(bottom[0].shape[0]):
            for j in range(bottom[0].shape[2]):
                for k in range(bottom[0].shape[3]):
                    label = int(bottom[1].data[i, 0, j, k])
                    bottom[0].diff[i, 0, j, k] = self.weight[label] * top[0].diff[i, 0, j, k]
                    # print self.blobs[0].diff[i, 0, :, :]
            print "\n"


class CenterPixelLayer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0], bottom[0].shape[1])

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[:, :, 127, 127]

    def backward(self, top, propagate_down, bottom):
        pass


class ValTestAccuracyLayer(caffe.Layer):
    """
    Rewrite Accuracy layer as a Python layer
    Use like this:
    layer {
        name: "accuracy"
        type: "Python"
        bottom: "pred"
        bottom: "label"
        top: "val-Accuracy"
        top: "val-TPR"
        top: "val-TNR"
        top: "val-VOE"
        top: "test-Accuracy"
        top: "test-TPR"
        top: "test-TNR"
        top: "test-VOE"
        include {
            phase: TEST
        }
        python_param {
            module: ""
            layer: "AccuracyLayer"
        }
    }
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two layer.bottoms'
        assert len(top) == 4, 'requires 3 layer.top'

    def reshape(self, bottom, top):
        # val
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
        # test
        top[4].reshape(1)
        top[5].reshape(1)
        top[6].reshape(1)
        top[7].reshape(1)

    def forward(self, bottom, top):
        # Renaming for clarity
        predictions = bottom[0].data
        # print 'predictions', predictions
        predictions = np.argmax(predictions, axis=1)  # make it labels instead of vectors of prob
        ground_truth = bottom[1].data
        # print 'ground_truth', ground_truth

        for k in [0, 1]:  # val, test
            tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
            half_count = predictions.shape[0] / 2
            for i in range(half_count):  # in batch
                j = k * half_count + i
                p = ground_truth[j] == 1
                t = predictions[j] == ground_truth[j]
                if t:
                    if p:
                        tp += 1
                    else:  # n
                        tn += 1
                else:  # f
                    if p:
                        fp += 1
                    else:  # n
                        fn += 1

            print tp, tn, fp, fn

            top[4 * k + 0].data[0] = 100 * (tp + tn) / len(ground_truth)  # acc
            top[4 * k + 1].data[0] = 0 if tp == 0 else 100 * tp / (tp + fn)  # tpr
            top[4 * k + 2].data[0] = 0 if tn == 0 else 100 * tn / (fp + tn)  # tnr
            top[4 * k + 3].data[0] = 100 * (1 - tp / (tp + fp + fn))  # voe

    def backward(self, top, propagate_down, bottom):
        pass


class AccuracySpatialLayer(caffe.Layer):
    """
    Rewrite Accuracy layer as a Python layer
    Use like this:
    layer {
        name: "accuracy"
        type: "Python"
        bottom: "pred"
        bottom: "label"
        top: "val-Accuracy"
        top: "val-TPR"
        top: "val-TNR"
        top: "val-VOE"
        include {
            phase: TEST
        }
        python_param {
            module: ""
            layer: "AccuracyLayer"
        }
    }
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two layer.bottoms'
        assert len(top) == 4, 'requires 3 layer.top'

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)

    def forward(self, bottom, top):
        # Renaming for clarity
        predictions = bottom[0].data
        # print 'predictions', predictions
        # predictions = np.argmax(predictions, axis=1)  # make it labels instead of vectors of prob
        ground_truth = bottom[1].data
        # print 'ground_truth pos count', np.sum(ground_truth)

        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for i in range(predictions.shape[0]):  # in batch
            for j in range(predictions.shape[2]):  # in w
                for k in range(predictions.shape[3]):  # in h
                    p = ground_truth[i, 0, j, k] == 1
                    t = predictions[i, 0, j, k] == ground_truth[i, 0, j, k]
                    if t:
                        if p:
                            tp += 1
                        else:  # n
                            tn += 1
                    else:  # f
                        if p:
                            fp += 1
                        else:  # n
                            fn += 1

            print tp, tn, fp, fn

            top[0].data[0] = 100 * (tp + tn) / (tp + tn + fp + fn)  # acc
            top[1].data[0] = 0 if tp == 0 else 100 * tp / (tp + fn)  # tpr
            top[2].data[0] = 0 if tn == 0 else 100 * tn / (fp + tn)  # tnr
            top[3].data[0] = 0 if tp == 0 else 100 * (1 - tp / (tp + fp + fn))  # voe

    def backward(self, top, propagate_down, bottom):
        pass


class AccuracyLayer(caffe.Layer):
    """
    Rewrite Accuracy layer as a Python layer
    Use like this:
    layer {
        name: "accuracy"
        type: "Python"
        bottom: "pred"
        bottom: "label"
        top: "val-Accuracy"
        top: "val-TPR"
        top: "val-TNR"
        top: "val-VOE"
        include {
            phase: TEST
        }
        python_param {
            module: ""
            layer: "AccuracyLayer"
        }
    }
    """

    def setup(self, bottom, top):
        assert len(bottom) == 2, 'requires two layer.bottoms'
        assert len(top) == 4, 'requires 3 layer.top'

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)

    def forward(self, bottom, top):
        # Renaming for clarity
        predictions = bottom[0].data
        # print 'predictions', predictions
        predictions = np.argmax(predictions, axis=1)  # make it labels instead of vectors of prob
        ground_truth = bottom[1].data
        # print 'ground_truth', ground_truth

        tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
        for i in range(predictions.shape[0]):  # in batch
            p = ground_truth[i] == 1
            t = predictions[i] == ground_truth[i]
            if t:
                if p:
                    tp += 1
                else:  # n
                    tn += 1
            else:  # f
                if p:
                    fp += 1
                else:  # n
                    fn += 1

            print tp, tn, fp, fn

            k = 0.00245  # pos/neg ratio in source

            top[0].data[0] = (tp + tn) / (tp + tn + fp + fn)  # acc balanced
            top[1].data[0] = 0 if tp == 0 else tp / (tp + fn)  # tpr
            top[2].data[0] = 0 if tn == 0 else tn / (fp + tn)  # tnr
            top[3].data[0] = 0 if tp + fn + fp == 0 else (fn + fp / k) / (
            tp + fn + fp / k)  # voe debalanced to original ratio

    def backward(self, top, propagate_down, bottom):
        pass


class CoupledDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) == 4, 'requires 4 layer.top'

        if hasattr(self, 'param_str') and self.param_str:
            params = json.loads(self.param_str)
        else:
            raise Exception("hasattr(self, 'param_str') and self.param_str")

        self.N = 32
        self.image_shape = (N, 1, 64, 64)
        self.label_shape = (N, 1)
        # self.map_size = N*1*64*64*4 * 10  #todo

        mean_file_path = params.get('mean_file', 1)
        source1_path = params.get('source1', 1)
        source2_path = params.get('source2', 1)
        self.db1 = lmdb.open(source1_path, readonly=True)
        self.db2 = lmdb.open(source2_path, readonly=True)
        self.mean = ndimage.imread(mean_file_path)

    def reshape(self, bottom, top):
        top[0].reshape(self.image_shape)
        top[1].reshape(self.label_shape)

    def forward(self, bottom, top):
        with self.db1.begin() as txn:
            cursor = txn.cursor()
            for i in range[0, N]:
                key, value = cursor[i]
                print key, value

    def backward(self, top, propagate_down, bottom):
        pass


class CentralPixelLayer(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) == 1, 'requires 1 layer.top'

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].shape[0], 1)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data[:, :, 31, 31]

    def backward(self, top, propagate_down, bottom):
        pass
