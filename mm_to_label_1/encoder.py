# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops
import logging

class Encoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', slice_stride=2, keep_prob=1.0,reuse=False):
        self.name = name
        self.reuse = reuse
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob

    def __call__(self, EC_input):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            EC_input = tf.nn.dropout(EC_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=EC_input, filters=self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=2 * self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(norm2)
            # pool1
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=4 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops._norm(conv3, self.is_training, self.norm)
                relu3 = ops.relu(norm3)
            # w/2,h/2
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=4 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv4')
                norm4 = ops._norm(conv4, self.is_training, self.norm)
                relu4 = ops.relu(norm4)
            with tf.variable_scope("conv5", reuse=self.reuse):
                conv5 = tf.layers.conv2d(inputs=relu4, filters=4 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv5')
                norm5 = ops._norm(conv5, self.is_training, self.norm)
                relu5 = tf.nn.relu(norm5)
            # pool2
            with tf.variable_scope("conv6", reuse=self.reuse):
                conv6 = tf.layers.conv2d(inputs=relu5, filters=8 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv6')
                norm6 = ops._norm(conv6, self.is_training, self.norm)
                relu6 = ops.relu(norm6)
            # w/4,h/4
            with tf.variable_scope("conv7", reuse=self.reuse):
                conv7 = tf.layers.conv2d(inputs=relu6, filters=8 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv7')
                norm7 = ops._norm(conv7, self.is_training, self.norm)
                output = ops.relu(norm7)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
