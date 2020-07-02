from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from fewshot.models.resnet_base import ResnetBase
from fewshot.models.model_factory import RegisterModel
from fewshot.utils.logger import get as get_logger

log = get_logger()


@RegisterModel("Encoder_g_gamma")
class Encoder_g_gamma(ResnetBase):
    def __call__(self,
               h,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top',
               h_size=512):
        self._ext_wts = ext_wts
        with tf.variable_scope('encoder_g_gamma', reuse=reuse):
            gamma = self.build_encoder_network(h, is_training, h_size)
        return gamma

    def build_encoder_network(self, h, is_training, h_size):
        config = self.config
        # x_shape = h.get_shape()
        # d = x_shape[1]
        # d = 512
        # d = 256
        # h_shape = h.get_shape()
        # h_size = 1
        # for ss in h_shape[1:]:
        #     h_size *= int(ss)
        d = h_size

        with tf.variable_scope('unit1'):
            out1 = self._fully_connected_encoder(h, d, d)
        out1 = out1 + h
        h = tf.nn.relu(out1)
        with tf.variable_scope('unit2'):
            out2 = self._fully_connected_encoder(h, d, d)
        out2 = out2 + h
        h = tf.nn.relu(out2)
        with tf.variable_scope('unit3'):
            out3 = self._fully_connected_encoder(h, d, d)
        out3 = out3 + h
        return tf.nn.relu(out3)

@RegisterModel("Encoder_g_beta")
class Encoder_g_beta(ResnetBase):
    def __call__(self,
               h,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top',
               h_size=512):
        self._ext_wts = ext_wts
        with tf.variable_scope('encoder_g_beta', reuse=reuse):
            beta = self.build_encoder_network(h, is_training, h_size)
        return beta

    def build_encoder_network(self, h, is_training, h_size):
        config = self.config
        # x_shape = h.get_shape()
        # d = x_shape[1]
        # d = 512
        # d = 256
        # h_shape = h.get_shape()
        # h_size = 1
        # for ss in h_shape[1:]:
        #     h_size *= int(ss)
        d = h_size

        with tf.variable_scope('unit1'):
            out1 = self._fully_connected_encoder(h, d, d)
        out1 = out1 + h
        h = tf.nn.relu(out1)
        with tf.variable_scope('unit2'):
            out2 = self._fully_connected_encoder(h, d, d)
        out2 = out2 + h
        h = tf.nn.relu(out2)
        with tf.variable_scope('unit3'):
            out3 = self._fully_connected_encoder(h, d, d)
        out3 = out3 + h
        return tf.nn.relu(out3)


@RegisterModel("Encoder_r")
class Encoder_r(ResnetBase):
    def __call__(self,
               h,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top',
               num_classes_a=64):
        self._ext_wts = ext_wts
        with tf.variable_scope('encoder_r', reuse=reuse):
            out = self.build_encoder_network(h, is_training, num_classes_a)
        return out

    def build_encoder_network(self, h, is_training, num_classes_a):
        config = self.config
        # x_shape = h.get_shape()
        # d = x_shape[1]
        # d = 200
        d = num_classes_a
        with tf.variable_scope('unit1'):
            out1 = self._fully_connected_encoder(h, d, d)
        h = tf.nn.relu(out1)
        with tf.variable_scope('unit2'):
            out2 = self._fully_connected_encoder(h, d, d)
        out2 = out2 + h
        h = tf.nn.relu(out2)
        with tf.variable_scope('unit3'):
            out3 = self._fully_connected_encoder(h, d, d)
        out3 = out3 + h
        return tf.nn.relu(out3)

@RegisterModel("Encoder_h")
class Encoder_h(ResnetBase):
    def __call__(self,
               h,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top',
               h_size=512):
        self._ext_wts = ext_wts
        with tf.variable_scope('encoder_h', reuse=reuse):
            out = self.build_encoder_network(h, is_training, h_size)
        return out

    def build_encoder_network(self, h, is_training, h_size):
        config = self.config
        x_shape = h.get_shape()
        # d = x_shape[1].value
        # d = 512
        # h_shape = h.get_shape()
        # h_size = 1
        # for ss in h_shape[1:]:
        #     h_size *= int(ss)
        d = h_size

        with tf.variable_scope('unit1'):
            out1 = self._fully_connected_encoder(h, d, d*2)
        h = tf.nn.relu(out1)
        with tf.variable_scope('unit2'):
            out2 = self._fully_connected_encoder(h, d, d)
        h = tf.nn.relu(out2)
        with tf.variable_scope('unit3'):
            out3 = self._fully_connected_encoder(h, d, d)
        h = tf.nn.relu(out3)
        with tf.variable_scope('unit4'):
            out4 = self._fully_connected_encoder(h, d, d)
        return tf.nn.sigmoid(out4)

@RegisterModel("Encoder_h_pre")
class Encoder_h_pre(ResnetBase):
    def __call__(self,
               h,
               is_training=True,
               ext_wts=None,
               reuse=None,
               slow_bn=False,
               exit='top',
               h_size=512):
        self._ext_wts = ext_wts
        with tf.variable_scope('encoder_h_pre', reuse=reuse):
            out = self.build_encoder_network(h, is_training, h_size)
        return out

    def build_encoder_network(self, h, is_training, h_size):
        config = self.config
        x_shape = h.get_shape()
        # d = x_shape[1].value
        # d = 512
        # d = 256
        # h_shape = h.get_shape()
        # h_size = 1
        # for ss in h_shape[1:]:
        #     h_size *= int(ss)
        d = h_size

        with tf.variable_scope('unit1'):
            out1 = self._fully_connected_encoder(h, d, d*2)
        h = tf.nn.relu(out1)
        with tf.variable_scope('unit2'):
            out2 = self._fully_connected_encoder(h, d, d)
        h = tf.nn.relu(out2)
        with tf.variable_scope('unit3'):
            out3 = self._fully_connected_encoder(h, d, d)
        h = tf.nn.relu(out3)
        with tf.variable_scope('unit4'):
            out4 = self._fully_connected_encoder(h, d, d)
        return tf.nn.sigmoid(out4)