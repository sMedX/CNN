# -*- coding: utf-8 -*-

import cupy
import chainer
import chainer.functions as F
import chainer.links as L

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

# Constant variables
N_LANDMARK = 68
IMG_SIZE = (224, 224)


def _disconnect(x):
    with chainer.no_backprop_mode():
        if isinstance(x, cupy.core.core.ndarray):
            return chainer.Variable(x).data

        return chainer.Variable(x.data).data


def copy_layers(src_model, dst_model,
                names=['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3']):
    for name in names:
        for s, d in zip(src_model[name].params(), dst_model[name].params()):
            d.data = s.data


class HyperFaceModel(chainer.Chain):

    def __init__(self, loss_weights=(1.0, 100.0, 20.0, 5.0, 0.3)):
        super(HyperFaceModel, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_a=L.Convolution2D(256, 512, 4, stride=4, pad=2),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_a=L.Convolution2D(512, 256, 2, stride=2, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv_all=L.Convolution2D(1280, 192, 1, stride=1, pad=0),
            fc_full=L.Linear(8 * 8 * 192, 3072),
            fc_detection1=L.Linear(3072, 512),
            fc_detection2=L.Linear(512, 2),
            fc_landmarks1=L.Linear(3072, 512),
            fc_landmarks2=L.Linear(512, 42),
            fc_visibility1=L.Linear(3072, 512),
            fc_visibility2=L.Linear(512, 21),
            fc_pose1=L.Linear(3072, 512),
            fc_pose2=L.Linear(512, 3),
            fc_gender1=L.Linear(3072, 512),
            fc_gender2=L.Linear(512, 2),
        )
        self.train = True
        self.report = True
        self.backward = True
        assert(len(loss_weights) == 5)
        self.loss_weights = loss_weights

    def __call__(self, x_img, t_detection=None, t_landmark=None,
                 t_visibility=None, t_pose=None, t_gender=None,
                 m_landmark=None, m_visibility=None, m_pose=None):
        # VGG
        h = F.relu(self.conv1_1(x_img))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h3 = F.relu(self.conv3_a(h))

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h4 = F.relu(self.conv4_a(h))

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.concat((h3, h4, h))

        # Fusion CNN
        h = F.relu(self.conv_all(h))  # conv_all
        h = F.relu(self.fc_full(h))  # fc_full
        with chainer.using_config('train', self.train):
            h = F.dropout(h)

        h_detection = F.relu(self.fc_detection1(h))
        with chainer.using_config('train', self.train):
            h_detection = F.dropout(h_detection)
        h_detection = self.fc_detection2(h_detection)
        h_landmark = F.relu(self.fc_landmarks1(h))
        with chainer.using_config('train', self.train):
            h_landmark = F.dropout(h_landmark)
        h_landmark = self.fc_landmarks2(h_landmark)
        h_visibility = F.relu(self.fc_visibility1(h))
        with chainer.using_config('train', self.train):
            h_visibility = F.dropout(h_visibility)
        h_visibility = self.fc_visibility2(h_visibility)
        h_pose = F.relu(self.fc_pose1(h))
        with chainer.using_config('train', self.train):
            h_pose = F.dropout(h_pose)
        h_pose = self.fc_pose2(h_pose)
        h_gender = F.relu(self.fc_gender1(h))
        with chainer.using_config('train', self.train):
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
            chainer.report({'conv1_1_w': {'weights': self.conv1_1.W},
                            'conv2_1_w': {'weights': self.conv2_1.W},
                            'conv3_1_w': {'weights': self.conv3_1.W},
                            'conv4_1_w': {'weights': self.conv4_1.W},
                            'conv5_1_w': {'weights': self.conv5_1.W}}, self)

        if self.backward:
            return loss
        else:
            return {'img': x_img, 'detection': h_detection,
                    'landmark': h_landmark, 'visibility': h_visibility,
                    'pose': h_pose, 'gender': h_gender}

class HyperFaceMenpoModel(chainer.Chain):

    def __init__(self):
        super(HyperFaceMenpoModel, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_a=L.Convolution2D(256, 512, 4, stride=4, pad=2),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_a=L.Convolution2D(512, 256, 2, stride=2, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv_all=L.Convolution2D(1280, 192, 1, stride=1, pad=0),

            fc_full=L.Linear(8 * 8 * 192, 3072),
            fc_landmarks1=L.Linear(3072, 1024),
            fc_landmarks2=L.Linear(1024, 68),
        )
        self.train = True
        self.report = True
        self.backward = True

    def __call__(self, x_img, t_landmark=None, m_landmark=None):
        # VGG
        h = F.relu(self.conv1_1(x_img))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h3 = F.relu(self.conv3_a(h))

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h4 = F.relu(self.conv4_a(h))

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.concat((h3, h4, h))

        # Fusion CNN
        h = F.relu(self.conv_all(h))  # conv_all
        h = F.relu(self.fc_full(h))  # fc_full
        with chainer.using_config('train', self.train):
            h = F.dropout(h)

        h_landmark = F.relu(self.fc_landmarks1(h))
        with chainer.using_config('train', self.train):
            h_landmark = F.dropout(h_landmark)

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

            # Loss
            loss_landmark = F.mean_squared_error(h_landmark, t_landmark)

            loss = loss_landmark

        if self.report:
            if self.backward:
                # Report losses
                chainer.report({'loss': loss}, self)

            # Report results
            predict_data = {'img': x_img, 'landmark': h_landmark}
            teacher_data = {'img': x_img, 'landmark': t_landmark}
            chainer.report({'predict': predict_data}, self)
            chainer.report({'teacher': teacher_data}, self)

            # Report layer weights
            chainer.report({'conv1_1_w': {'weights': self.conv1_1.W},
                            'conv2_1_w': {'weights': self.conv2_1.W},
                            'conv3_1_w': {'weights': self.conv3_1.W},
                            'conv4_1_w': {'weights': self.conv4_1.W},
                            'conv5_1_w': {'weights': self.conv5_1.W}}, self)

        if self.backward:
            return loss
        else:
            return {'img': x_img, 'landmark': h_landmark}


class RCNNFaceModel(chainer.Chain):

    def __init__(self):
        super(RCNNFaceModel, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            fc6=L.Linear(8 * 8 * 512, 4096),
            fc7=L.Linear(4096, 512),
            fc8=L.Linear(512, 2),
        )
        self.train = True

    def __call__(self, x_img, t_detection, **others):
        # VGG
        h = F.relu(self.conv1_1(x_img))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        # fc6
        h = F.relu(self.fc6(h))
        with chainer.using_config('train', self.train):
            h = F.dropout(h)

        # fc7
        h = F.relu(self.fc7(h))
        with chainer.using_config('train', self.train):
            h = F.dropout(h)

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
        chainer.report({'conv1_1_w': {'weights': self.conv1_1.W},
                        'conv2_1_w': {'weights': self.conv2_1.W},
                        'conv3_1_w': {'weights': self.conv3_1.W},
                        'conv4_1_w': {'weights': self.conv4_1.W},
                        'conv5_1_w': {'weights': self.conv5_1.W}}, self)

        return loss
