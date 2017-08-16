# -*- coding: utf-8 -*-

import numpy as np
import cupy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.serializers import npz

from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer.functions.pooling.average_pooling_2d import average_pooling_2d
from resnet import BuildingBlock, transfer_block, global_average_pooling_2d

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Constant variables
N_LANDMARK = 21
IMG_SIZE = (227, 227)


def _disconnect(x):
    with chainer.no_backprop_mode():
        if isinstance(x, cupy.core.core.ndarray):
            return chainer.Variable(x).data

        return chainer.Variable(x.data).data


def copy_layers_from_caffemodel(src, dst):
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    transfer_block(src, dst.res5, ['5a', '5b', '5c'])


def copy_layers_from_pretrainedmodel(src, dst):
    dst.conv1.copyparams(src.conv1)
    dst.bn1.copyparams(src.bn1)
    dst.res2.copyparams(src.res2)
    dst.res3.copyparams(src.res3)
    dst.res4.copyparams(src.res4)
    dst.res5.copyparams(src.res5)


class HyperFaceModel(chainer.Chain):

    def __init__(self, loss_weights=(1.0, 100.0, 20.0, 5.0, 0.3), n_resnet_layers=50):
        super(HyperFaceModel, self).__init__()

        if n_resnet_layers == 50:
            resnet_block = [3, 4, 6, 3]
        elif n_resnet_layers == 101:
            resnet_block = [3, 4, 23, 3]
        elif n_resnet_layers == 152:
            resnet_block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            # ResNet
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3)
            self.bn1 = BatchNormalization(64)
            self.res2 = BuildingBlock(resnet_block[0], 64, 64, 256, 1)
            self.res3 = BuildingBlock(resnet_block[1], 256, 128, 512, 2)
            self.res4 = BuildingBlock(resnet_block[2], 512, 256, 1024, 2)
            self.res5 = BuildingBlock(resnet_block[3], 1024, 512, 2048, 2)
            # Fusion CNN
            self.conv_all = L.Convolution2D(2048, 192, 1, stride=1, pad=0)
            self.fc_full = L.Linear(15 * 15 * 192, 3072)
            self.fc_detection1 = L.Linear(3072, 512)
            self.fc_detection2 = L.Linear(512, 2)
            self.fc_landmarks1 = L.Linear(3072, 512)
            self.fc_landmarks2 = L.Linear(512, 42)
            self.fc_visibility1 = L.Linear(3072, 512)
            self.fc_visibility2 = L.Linear(512, 21)
            self.fc_pose1 = L.Linear(3072, 512)
            self.fc_pose2 = L.Linear(512, 3)
            self.fc_gender1 = L.Linear(3072, 512)
            self.fc_gender2 = L.Linear(512, 2)

        self.train = True
        self.report = True
        self.backward = True
        assert(len(loss_weights) == 5)
        self.loss_weights = loss_weights

    def __call__(self, x_img, t_detection=None, t_landmark=None,
                 t_visibility=None, t_pose=None, t_gender=None,
                 m_landmark=None, m_visibility=None, m_pose=None):
        # ResNet
        h = self.bn1(self.conv1(x_img))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)

        # Fusion CNN
        with chainer.using_config('train', self.train):
            h = F.relu(self.conv_all(h))
            h = F.relu(self.fc_full(h))
            h = F.dropout(h)

            h_detection = F.relu(self.fc_detection1(h))
            h_detection = F.dropout(h_detection)
            h_detection = self.fc_detection2(h_detection)
            h_landmark = F.relu(self.fc_landmarks1(h))
            h_landmark = F.dropout(h_landmark)
            h_landmark = self.fc_landmarks2(h_landmark)
            h_visibility = F.relu(self.fc_visibility1(h))
            h_visibility = F.dropout(h_visibility)
            h_visibility = self.fc_visibility2(h_visibility)
            h_pose = F.relu(self.fc_pose1(h))
            h_pose = F.dropout(h_pose)
            h_pose = self.fc_pose2(h_pose)
            h_gender = F.relu(self.fc_gender1(h))
            h_gender = F.dropout(h_gender)
            h_gender = self.fc_gender2(h_gender)

        # Mask and Loss
        if self.backward:
            # Landmark masking with visibility
            m_landmark_ew = F.stack((t_visibility, t_visibility), axis=2)
            m_landmark_ew = F.reshape(m_landmark_ew, (-1, N_LANDMARK * 2))

            # Masking
            h_landmark *= _disconnect(m_landmark)
            t_landmark *= _disconnect(m_landmark)
            h_landmark *= _disconnect(m_landmark_ew)
            t_landmark *= _disconnect(m_landmark_ew)
            h_visibility *= _disconnect(m_visibility)
            t_visibility *= _disconnect(m_visibility)
            h_pose *= _disconnect(m_pose)
            t_pose *= _disconnect(m_pose)

            # Loss
            loss_detection = F.softmax_cross_entropy(h_detection, t_detection)
            loss_landmark = F.mean_squared_error(h_landmark, t_landmark)
            loss_visibility = F.mean_squared_error(h_visibility, t_visibility)
            loss_pose = F.mean_squared_error(h_pose, t_pose)
            loss_gender = F.softmax_cross_entropy(h_gender, t_gender)

            # Loss scaling
            loss_detection *= self.loss_weights[0]
            loss_landmark *= self.loss_weights[1]
            loss_visibility *= self.loss_weights[2]
            loss_pose *= self.loss_weights[3]
            loss_gender *= self.loss_weights[4]

            loss = (loss_detection + loss_landmark + loss_visibility +
                    loss_pose + loss_gender)

        # Prediction (the same shape as t_**, and [0:1])
        h_detection = F.softmax(h_detection)[:, 1]  # ([[y, n]] -> [d])
        h_gender = F.softmax(h_gender)[:, 1]  # ([[m, f]] -> [g])

        if self.report:
            if self.backward:
                # Report losses
                chainer.report({'loss': loss,
                                'loss_detection': loss_detection,
                                'loss_landmark': loss_landmark,
                                'loss_visibility': loss_visibility,
                                'loss_pose': loss_pose,
                                'loss_gender': loss_gender}, self)

            # Report results
            predict_data = {'img': x_img, 'detection': h_detection,
                            'landmark': h_landmark, 'visibility': h_visibility,
                            'pose': h_pose, 'gender': h_gender}
            teacher_data = {'img': x_img, 'detection': t_detection,
                            'landmark': t_landmark, 'visibility': t_visibility,
                            'pose': t_pose, 'gender': t_gender}
            chainer.report({'predict': predict_data}, self)
            chainer.report({'teacher': teacher_data}, self)

            # Report layer weights
            # chainer.report({'conv1_w': {'weights': self.conv1.W},
            #                 'conv2_w': {'weights': self.conv2.W},
            #                 'conv3_w': {'weights': self.conv3.W},
            #                 'conv4_w': {'weights': self.conv4.W},
            #                 'conv5_w': {'weights': self.conv5.W}}, self)

        if self.backward:
            return loss
        else:
            return {'img': x_img, 'detection': h_detection,
                    'landmark': h_landmark, 'visibility': h_visibility,
                    'pose': h_pose, 'gender': h_gender}


class RCNNFaceModel(chainer.Chain):

    def __init__(self, n_resnet_layers=50):
        super(RCNNFaceModel, self).__init__()

        if n_resnet_layers == 50:
            resnet_block = [3, 4, 6, 3]
        elif n_resnet_layers == 101:
            resnet_block = [3, 4, 23, 3]
        elif n_resnet_layers == 152:
            resnet_block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))

        with self.init_scope():
            # ResNet
            self.conv1 = L.Convolution2D(3, 64, 7, 2, 3)
            self.bn1 = BatchNormalization(64)
            self.res2 = BuildingBlock(resnet_block[0], 64, 64, 256, 1)
            self.res3 = BuildingBlock(resnet_block[1], 256, 128, 512, 2)
            self.res4 = BuildingBlock(resnet_block[2], 512, 256, 1024, 2)
            self.res5 = BuildingBlock(resnet_block[3], 1024, 512, 2048, 2)
            # RCNN
            self.fc6 = L.Linear(2048, 1024)
            self.fc7 = L.Linear(1024, 512)
            self.fc8 = L.Linear(512, 2)

        self.train = True

    def __call__(self, x_img, t_detection, **others):
        # ResNet
        h = self.bn1(self.conv1(x_img))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = global_average_pooling_2d(h)

        # RCNN
        with chainer.using_config('train', self.train):
            h = F.dropout(F.relu(self.fc6(h)))  # fc6
            h = F.dropout(F.relu(self.fc7(h)))  # fc7
        h_detection = self.fc8(h)  # fc8

        # Loss
        loss = F.softmax_cross_entropy(h_detection, t_detection)

        chainer.report({'loss': loss}, self)

        # Prediction
        h_detection = F.argmax(h_detection, axis=1)

        # Report results
        predict_data = {'img': x_img, 'detection': h_detection}
        teacher_data = {'img': x_img, 'detection': t_detection}
        chainer.report({'predict': predict_data}, self)
        chainer.report({'teacher': teacher_data}, self)

        # Report layer weights
        # chainer.report({'conv1_w': {'weights': self.conv1.W},
        #                 'conv2_w': {'weights': self.conv2.W},
        #                 'conv3_w': {'weights': self.conv3.W},
        #                 'conv4_w': {'weights': self.conv4.W},
        #                 'conv5_w': {'weights': self.conv5.W}}, self)

        return loss
