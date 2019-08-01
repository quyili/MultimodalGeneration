# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class FeatureDiscriminator:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', slice_stride=2, keep_prob=1.0):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.ngf = ngf
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob

    def __call__(self, FD_input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            FD_input = tf.nn.dropout(FD_input, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=FD_input, filters=self.ngf, kernel_size=7,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=self.ngf, kernel_size=5,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops._norm(conv3, self.is_training, self.norm)
                relu3 = ops.relu(norm3)
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv4')
                norm4 = ops._norm(conv4, self.is_training, self.norm)
                relu4 = ops.relu(norm4)
            with tf.variable_scope("conv5", reuse=self.reuse):
                output = tf.layers.conv2d(inputs=relu4, filters=1, kernel_size=3, strides=1,
                                          padding="SAME",
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(
                                              mean=0.0, stddev=0.02, dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0), name='conv5')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output