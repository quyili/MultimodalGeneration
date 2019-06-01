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

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        input1, input2 = input
        with tf.variable_scope(self.name, reuse=self.reuse):
            input1 = tf.nn.dropout(input1, keep_prob=self.keep_prob)
            input2 = tf.nn.dropout(input2, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1_1", reuse=self.reuse):
                conv1_1 = tf.layers.conv2d(inputs=input1, filters=2 * self.ngf, kernel_size=5,
                                           strides=self.slice_stride,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv1_1')
                norm1_1 = ops._norm(conv1_1, self.is_training, self.norm)
            with tf.variable_scope("conv1_2", reuse=self.reuse):
                conv1_2 = tf.layers.conv2d(inputs=input2, filters=2 * self.ngf, kernel_size=5, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=0.0, stddev=0.02, dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv1_2')
                norm1_2 = ops._norm(conv1_2, self.is_training, self.norm)
            with tf.variable_scope("add1", reuse=self.reuse):
                add1 = ops.relu(tf.add(norm1_1, norm1_2))
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=add1, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=4 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=0.0, stddev=0.02, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                norm3 = ops._norm(conv3, self.is_training, self.norm)
                relu3 = ops.relu(norm3)
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=4 * self.ngf, kernel_size=3, strides=1,
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
