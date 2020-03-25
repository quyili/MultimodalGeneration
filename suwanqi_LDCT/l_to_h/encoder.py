# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class Encoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', slice_stride=2, keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.slice_stride = slice_stride
        self.keep_prob = keep_prob
        self.output_channl = output_channl

    def __call__(self, EC_input, wave1):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """

        with tf.variable_scope(self.name):
            EC_input = tf.nn.dropout(EC_input, keep_prob=self.keep_prob)
            wave1 = tf.nn.dropout(wave1, keep_prob=self.keep_prob)
            shape1 = EC_input.get_shape().as_list()
            shape2 = wave1.get_shape().as_list()
            with tf.variable_scope("deconv_wave1", reuse=self.reuse):
                resize1 = ops.uk_resize(wave1, reuse=self.reuse, name='resize1',
                                        output_size=[shape1[1], shape1[2]])
            with tf.variable_scope("concat", reuse=self.reuse):
                concat = tf.reshape(tf.concat([EC_input, resize1], axis=-1),
                                    [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]])
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=concat, filters=self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                # norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(conv1)
                # print("conv1:", relu1.get_shape().as_list())
            with tf.variable_scope("conv2", reuse=self.reuse):
                conv2 = tf.layers.conv2d(inputs=relu1, filters=self.ngf, kernel_size=3, strides=1, padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2')
                # norm2 = ops._norm(conv2, self.is_training, self.norm)
                relu2 = ops.relu(conv2)
                # print("conv2:", relu2.get_shape().as_list())
            # pool1
            with tf.variable_scope("conv3", reuse=self.reuse):
                conv3 = tf.layers.conv2d(inputs=relu2, filters=2 * self.ngf, kernel_size=3,
                                         strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv3')
                # norm3 = ops._norm(conv3, self.is_training, self.norm)
                relu3 = ops.relu(conv3)
                # print("conv3:", conv3.get_shape().as_list())
            with tf.variable_scope("conv4", reuse=self.reuse):
                conv4 = tf.layers.conv2d(inputs=relu3, filters=2 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv4')
                # norm4 = ops._norm(conv4, self.is_training, self.norm)
                relu4 = ops.relu(conv4)
                # print("conv4:", relu4.get_shape().as_list())
            with tf.variable_scope("conv5", reuse=self.reuse):
                conv5 = tf.layers.conv2d(inputs=relu4, filters=4 * self.ngf, kernel_size=3, strides=self.slice_stride,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv5')
                # norm5 = ops._norm(conv5, self.is_training, self.norm)
                relu5 = tf.nn.relu(conv5)
                # print("conv5:", relu5.get_shape().as_list())
            """
            decoder
            """
            with tf.variable_scope("conv6", reuse=self.reuse):
                conv6 = tf.layers.conv2d(inputs=relu5, filters=4 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                         dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0),
                                         name='conv6')
                # add1_norm1 = ops._norm(add1_conv1, self.is_training, self.norm)
                relu6 = ops.relu(conv6)
            with tf.variable_scope("deconv7_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu6, reuse=self.reuse, name='resize1')
                deconv7_r = tf.layers.conv2d(inputs=resize1, filters=2 * self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                             dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv7_r')
                # deconv1_norm1_r = ops._norm(deconv1_r, self.is_training, self.norm)
                deconv7_r += relu4
                deconv7 = ops.relu(deconv7_r)
                # print("deconv1:", deconv1.get_shape().as_list())
                # print("concat1:", concat1.get_shape().as_list())
                # print("add1_conv1:", add1_relu1.get_shape().as_list())
            with tf.variable_scope("conv8", reuse=self.reuse):
                conv8 = tf.layers.conv2d(inputs=deconv7, filters=2 * self.ngf, kernel_size=3,
                                         strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                         dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0),
                                         name='conv8')
                # add1_norm2 = ops._norm(add1_conv2, self.is_training, self.norm)
                relu8 = ops.relu(conv8)
                # print("add1_conv2:", add1_relu2.get_shape().as_list())
            with tf.variable_scope("deconv9_r", reuse=self.reuse):
                resize2 = ops.uk_resize(relu8, reuse=self.reuse, name='resize2')
                deconv9_r = tf.layers.conv2d(inputs=resize2, filters=self.ngf, kernel_size=3, strides=1,
                                             padding="SAME",
                                             activation=None,
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                             dtype=tf.float32),
                                             bias_initializer=tf.constant_initializer(0.0),
                                             name='deconv9_r')
                # deconv2_norm1_r = ops._norm(deconv2_r, self.is_training, self.norm)
                deconv9_r += relu2
                deconv9 = ops.relu(deconv9_r)
                # print("concat2:", concat2.get_shape().as_list())
            with tf.variable_scope("conv10", reuse=self.reuse):
                conv10 = tf.layers.conv2d(inputs=deconv9, filters=self.ngf, kernel_size=3,
                                          strides=1,
                                          padding="SAME",
                                          activation=None,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                          dtype=tf.float32),
                                          bias_initializer=tf.constant_initializer(0.0),
                                          name='conv10')
                # add2_norm1 = ops._norm(add2_conv1, self.is_training, self.norm)
                relu10 = ops.relu(conv10)

            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv = tf.layers.conv2d(inputs=relu10, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                            dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv')

                output = tf.nn.tanh(lastconv)
                # print("output:", output.get_shape().as_list())
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
