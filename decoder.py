# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class Decoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance',
                 skip_type="concat", slice_stride=2, keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.slice_stride = slice_stride
        self.skip_type = skip_type
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, input):
        """
        Args:
          input: batch_size x width x height x N
        Returns:
          output: same size as input
        """
        input1, input2 = input
        with tf.variable_scope(self.name, reuse=self.reuse):
            input1 = tf.nn.dropout(input1, keep_prob=self.keep_prob)
            input2 = tf.nn.dropout(input2, keep_prob=self.keep_prob)
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=input2, filters=6 * self.ngf, kernel_size=3, strides=1,
                                          padding="SAME",
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(
                                              mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
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
                                                 mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv1_r')
                deconv1_norm1_r = ops._norm(deconv1_r, self.is_training, self.norm)
            with tf.variable_scope("deconv1_t", reuse=self.reuse):
                deconv1_t = tf.layers.conv2d_transpose(inputs=relu1, filters=4 * self.ngf, kernel_size=3,
                                                       strides=self.slice_stride,
                                                       padding="SAME",
                                                       activation=None,
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                           dtype=tf.float32),
                                                       bias_initializer=tf.constant_initializer(0.0),
                                                       name='deconv1_t')
                deconv1_norm1_t = ops._norm(deconv1_t, self.is_training, self.norm)
            with tf.variable_scope("add1", reuse=self.reuse):
                add1 = ops.relu(tf.add(deconv1_norm1_r * 0.8, deconv1_norm1_t * 0.2))
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
            with tf.variable_scope("concat1", reuse=self.reuse):
                shape1 = input1.get_shape().as_list()
                concat1 = tf.reshape(tf.concat([add1_relu1, input1], axis=-1),
                                     [shape1[0], shape1[1], shape1[2], 2 * shape1[3]])
                concat1_conv = tf.layers.conv2d(inputs=concat1, filters=4 * self.ngf, kernel_size=3, strides=1,
                                                padding="SAME",
                                                activation=None,
                                                kernel_initializer=tf.random_normal_initializer(
                                                    mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                                    dtype=tf.float32),
                                                bias_initializer=tf.constant_initializer(0.0),
                                                name='concat1_conv')
                concat1_norm2 = ops._norm(concat1_conv, self.is_training, self.norm)
                concat1_relu2 = ops.relu(concat1_norm2)

            with tf.variable_scope("deconv2_r", reuse=self.reuse):
                resize2 = ops.uk_resize(concat1_relu2, reuse=self.reuse, name='resize1')
                deconv2_r = tf.layers.conv2d(inputs=resize2, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(
                                                 mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                 dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv2_r')
                deconv2_norm1_r = ops._norm(deconv2_r, self.is_training, self.norm)
            with tf.variable_scope("deconv2_t", reuse=self.reuse):
                deconv2_t = tf.layers.conv2d_transpose(inputs=concat1_relu2, filters=2 * self.ngf,
                                                       kernel_size=3,
                                                       strides=self.slice_stride,
                                                       padding="SAME",
                                                       activation=None,
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           mean=1.0 / (9.0 * 4 * self.ngf), stddev=0.000001,
                                                           dtype=tf.float32),
                                                       bias_initializer=tf.constant_initializer(0.0),
                                                       name='deconv2_t')
                deconv2_norm1_t = ops._norm(deconv2_t, self.is_training, self.norm)
            with tf.variable_scope("add2", reuse=self.reuse):
                add2 = ops.relu(tf.add(deconv2_norm1_r * 0.8, deconv2_norm1_t * 0.2))
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
                                              mean=1.0 / (9.0 * 2*self.ngf), stddev=0.000001, dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0), name='conv2')
                norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(norm2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                output = tf.layers.conv2d(inputs=relu2, filters=self.output_channl, kernel_size=3, strides=1,
                                          padding="SAME",
                                          activation=tf.nn.sigmoid,
                                          kernel_initializer=tf.random_normal_initializer(
                                              mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0), name='lastconv')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
