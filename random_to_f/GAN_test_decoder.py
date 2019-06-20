# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops


class GDecoder:
    def __init__(self, name, ngf=64, is_training=True, norm='instance', slice_stride=2, keep_prob=1.0, output_channl=1):
        self.name = name
        self.reuse = False
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
            with tf.variable_scope("dense0", reuse=self.reuse):
                dense0 = tf.layers.dense(DC_input, units=DC_input.get_shape().as_list()[0] * 6 * 5 * self.ngf,
                                         name="dense0")
            with tf.variable_scope("dense1", reuse=self.reuse):
                dense1 = tf.layers.dense(dense0, units=DC_input.get_shape().as_list()[0] * 6 * 5 * 12 * self.ngf,
                                         name="dense0")
                dense1 = tf.reshape(dense1, shape=[DC_input.get_shape().as_list()[0], 6, 5, 12 * self.ngf])
            # 6,5
            with tf.variable_scope("conv0_1", reuse=self.reuse):
                conv0_1 = tf.layers.conv2d(inputs=dense1, filters=12 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 12 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_1')
                norm0_1 = ops._norm(conv0_1, self.is_training, self.norm)
                relu0_1 = ops.relu(norm0_1)
            # 6,5
            with tf.variable_scope("deconv0_1_r", reuse=self.reuse):
                resize0_1 = ops.uk_resize(relu0_1, reuse=self.reuse, output_size=[12, 9], name='resize')
                deconv0_1_r = tf.layers.conv2d(inputs=resize0_1, filters=8 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 12 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_1_r')
                deconv0_1_norm1_r = ops._norm(deconv0_1_r, self.is_training, self.norm)
                deconv0_1_relu1 = ops.relu(deconv0_1_norm1_r)
            # 12,9
            with tf.variable_scope("conv0_2", reuse=self.reuse):
                conv0_2 = tf.layers.conv2d(inputs=deconv0_1_relu1, filters=8 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_2')
                norm0_2 = ops._norm(conv0_2, self.is_training, self.norm)
                relu0_2 = ops.relu(norm0_2)
            # 12,9
            with tf.variable_scope("deconv0_2_r", reuse=self.reuse):
                resize0_2 = ops.uk_resize(relu0_2, reuse=self.reuse, output_size=[23, 18], name='resize')
                deconv0_2_r = tf.layers.conv2d(inputs=resize0_2, filters=6 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 8 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_2_r')
                deconv0_2_norm1_r = ops._norm(deconv0_2_r, self.is_training, self.norm)
                deconv0_2_relu1 = ops.relu(deconv0_2_norm1_r)
            # 23, 18
            with tf.variable_scope("conv0_3", reuse=self.reuse):
                conv0_3 = tf.layers.conv2d(inputs=deconv0_2_relu1, filters=6 * self.ngf, kernel_size=3, strides=1,
                                           padding="SAME",
                                           activation=None,
                                           kernel_initializer=tf.random_normal_initializer(
                                               mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                               dtype=tf.float32),
                                           bias_initializer=tf.constant_initializer(0.0), name='conv0_3')
                norm0_3 = ops._norm(conv0_3, self.is_training, self.norm)
                relu0_3 = ops.relu(norm0_3)
            # 23, 18
            with tf.variable_scope("deconv0_3_r", reuse=self.reuse):
                resize0_3 = ops.uk_resize(relu0_3, reuse=self.reuse, name='resize')
                deconv0_3_r = tf.layers.conv2d(inputs=resize0_3, filters=6 * self.ngf, kernel_size=3, strides=1,
                                               padding="SAME",
                                               activation=None,
                                               kernel_initializer=tf.random_normal_initializer(
                                                   mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                   dtype=tf.float32),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               name='deconv0_3_r')
                deconv0_3_norm1_r = ops._norm(deconv0_3_r, self.is_training, self.norm)
            with tf.variable_scope("deconv0_3_t", reuse=self.reuse):
                deconv0_3_t = tf.layers.conv2d_transpose(inputs=relu0_3, filters=6 * self.ngf, kernel_size=3,
                                                         strides=self.slice_stride,
                                                         padding="SAME",
                                                         activation=None,
                                                         kernel_initializer=tf.random_normal_initializer(
                                                             mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                                             dtype=tf.float32),
                                                         bias_initializer=tf.constant_initializer(0.0),
                                                         name='deconv0_3_t')
                deconv0_3_norm1_t = ops._norm(deconv0_3_t, self.is_training, self.norm)
            with tf.variable_scope("add0", reuse=self.reuse):
                add0 = ops.relu(tf.add(deconv0_3_norm1_r * 0.8, deconv0_3_norm1_t * 0.2))
            # 46, 36
            with tf.variable_scope("conv1", reuse=self.reuse):
                conv1 = tf.layers.conv2d(inputs=add0, filters=6 * self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 6 * self.ngf), stddev=0.000001,
                                             dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv1')
                norm1 = ops._norm(conv1, self.is_training, self.norm)
                relu1 = ops.relu(norm1)
            with tf.variable_scope("deconv1_r", reuse=self.reuse):
                resize1 = ops.uk_resize(relu1, reuse=self.reuse, name='resize')
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
                resize2 = ops.uk_resize(add1_relu2, reuse=self.reuse, name='resize')
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
                deconv2_t = tf.layers.conv2d_transpose(inputs=add1_relu2, filters=2 * self.ngf,
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

            with tf.variable_scope("conv2_1", reuse=self.reuse):
                conv2_1 = tf.layers.conv2d(inputs=add2_relu2, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2_1')
                norm2_1 = ops._norm(conv2_1, self.is_training, self.norm)
                relu2_1 = ops.relu(norm2_1)
            with tf.variable_scope("lastconv_1", reuse=self.reuse):
                lastconv_1 = tf.layers.conv2d(inputs=relu2_1, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv_1')
                lastnorm_1 = ops._norm(lastconv_1, self.is_training, self.norm)
                output_1 = tf.nn.sigmoid(lastnorm_1)

            with tf.variable_scope("conv2_2", reuse=self.reuse):
                conv2_2 = tf.layers.conv2d(inputs=add2_relu2, filters=self.ngf, kernel_size=3, strides=1,
                                         padding="SAME",
                                         activation=None,
                                         kernel_initializer=tf.random_normal_initializer(
                                             mean=1.0 / (9.0 * 2 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                         bias_initializer=tf.constant_initializer(0.0), name='conv2_2')
                norm2_2 = ops._norm(conv2_2, self.is_training, self.norm)
                relu2_2 = ops.relu(norm2_2)
            with tf.variable_scope("lastconv", reuse=self.reuse):
                lastconv_2 = tf.layers.conv2d(inputs=relu2_2, filters=self.output_channl, kernel_size=3, strides=1,
                                            padding="SAME",
                                            activation=None,
                                            kernel_initializer=tf.random_normal_initializer(
                                                mean=1.0 / (9.0 * self.ngf), stddev=0.000001, dtype=tf.float32),
                                            bias_initializer=tf.constant_initializer(0.0), name='lastconv_2')
                lastnorm_2 = ops._norm(lastconv_2, self.is_training, self.norm)
                output_2 = tf.nn.sigmoid(lastnorm_2)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output_1,output_2
