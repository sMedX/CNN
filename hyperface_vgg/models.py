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
N_AFLW_LANDMARK = 42
N_MENPO_LANDMARK = 68
IMG_SIZE = (227, 227)


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

    def __init__(self, menpo_dataset, loss_weights=(1.0, 100.0, 20.0, 5.0, 0.3)):
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

            fc_menpo_landmark1=L.Linear(3072, 512),
            fc_menpo_landmark2=L.Linear(512, N_MENPO_LANDMARK * 2),
        )
        self.menpo_dataset = menpo_dataset
        self.train = True
        self.report = True
        self.backward = True
        assert(len(loss_weights) == 5)
        self.loss_weights = loss_weights

    def __call__(self, x_img, t_menpo_landmark=None, m_menpo_landmark=None):
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

        h_menpo_landmark = F.relu(self.fc_menpo_landmark1(h))
        with chainer.using_config('train', self.train):
            h_landmark = F.dropout(h_menpo_landmark)
        h_menpo_landmark = self.fc_menpo_landmark2(h_menpo_landmark)

        # Mask and Loss
        if self.backward:
            # Landmark masking with visibility
            h_menpo_landmark *= _disconnect(m_menpo_landmark)
            t_menpo_landmark *= _disconnect(m_menpo_landmark)

            # Loss
            loss_landmark = F.mean_squared_error(
                h_menpo_landmark, t_menpo_landmark)

            loss = loss_landmark

        if self.report:
            if self.backward:
                # Report losses
                chainer.report({'loss': loss}, self)

            # Report results
            predict_data = {'img': x_img, 'menpo_landmark': h_menpo_landmark}
            teacher_data = {'img': x_img, 'menpo_landmark': t_menpo_landmark}
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
            return {'img': x_img, 'menpo_landmark': h_menpo_landmark}


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
