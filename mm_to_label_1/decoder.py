# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops
import logging

class Decoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', slice_stride=2, keep_prob=1.0, output_channl=1,reuse=False):
        self.name = name
        self.reuse = reuse
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, DC_input):
        """
        Args:
          input: batch_size x width x height x N
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            DC_input = tf.nn.dropout(DC_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=DC_input, filters=8 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                             dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu1, reuse=self.reuse, name='resize1')
                deconv1_r = tf.layers.conv2d(inputs=resize1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1_norm1_r = ops._norm(deconv1_r, self.is_training, self.norm)
                add1 = ops.relu(deconv1_norm1_r)
            with tf.variable_scope("add1_conv1", reuse=self.reuse):
                add1_conv1 = tf.layers.conv2d(inputs=add1, filters=4 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv1')
                add1_norm1 = ops._norm(add1_conv1, self.is_training, self.norm)
                add1_relu1 = ops.relu(add1_norm1)
            with tf.variable_scope("add1_conv2", reuse=self.reuse):
                add1_conv2 = tf.layers.conv2d(inputs=add1_relu1, filters=4 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add1_conv2')
                add1_norm2 = ops._norm(add1_conv2, self.is_training, self.norm)
                add1_relu2 = ops.relu(add1_norm2)
            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(add1_relu2, reuse=self.reuse, name='resize1')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2_norm1_r = ops._norm(deconv2_r, self.is_training, self.norm)
                add2 = ops.relu(deconv2_norm1_r)
            with tf.variable_scope("add2_conv1", reuse=self.reuse):
                add2_conv1 = tf.layers.conv2d(inputs=add2, filters=2 * self.ngf, kernel_size=3,
                                              strides=1,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                  dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='add2_conv1')
                add2_norm1 = ops._norm(add2_conv1, self.is_training, self.norm)
                add2_relu1 = ops.relu(add2_norm1)
            with tf.variable_scope("add2_conv2", reuse=self.reuse):
                add2_conv = tf.layers.conv2d(inputs=add2_relu1, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='add2_conv2')
                add2_norm2 = ops._norm(add2_conv, self.is_training, self.norm)
                add2_relu2 = ops.relu(add2_norm2)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=add2_relu2, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=relu2, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')
                lastnorm = ops._norm(lastconv, self.is_training, self.norm)
                output = tf.nn.sigmoid(lastnorm)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
