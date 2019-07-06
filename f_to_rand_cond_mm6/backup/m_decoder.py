# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class MDecoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
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
            with tf.variable_scope("conv0_1", reuse=self.reuse):
                conv0_1 = tf.layers.conv2d(inputs=DC_input, filters=2 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_1')
                norm0_1 = ops._norm(conv0_1, self.is_training, self.norm)
                relu0_1 = ops.relu(norm0_1)
            with tf.variable_scope("conv0_2", reuse=self.reuse):
                conv0_2 = tf.layers.conv2d(inputs=DC_input, filters=2 * self.ngf, kernel_size=5, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_2')
                norm0_2 = ops._norm(conv0_2, self.is_training, self.norm)
                relu0_2 = ops.relu(norm0_2)
            with tf.variable_scope("concat0", reuse=self.reuse):
                shape = DC_input.get_shape().as_list()
                concat0 = tf.reshape(tf.concat([relu0_1, relu0_2], axis=-1),
                                     shape=[shape[0], shape[1], shape[2], 4 * self.ngf])
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=concat0, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001,
                                             dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=self.ngf, kernel_size=3, strides=1,
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
