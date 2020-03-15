# _*_ coding:utf-8 _*_
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import ops


class Encoder:
    def __init__(self, name, ngf=96, keep_prob=1.0, kernel_size=[5, 5],
                 conv_stride=1, initial_std=0.01, last_activation_fn="sigmoid"):
        self.name = name
        self.reuse = False
        self.filter_size = ngf
        self.conv_stride = conv_stride
        self.keep_prob = keep_prob
        self.initial_std = initial_std
        self.kernel_size = kernel_size
        self.last_activation_fn = last_activation_fn

    def __call__(self, in_image):
        with tf.variable_scope(self.name, reuse=self.reuse):
            """
            encoder
            """
            in_image = tf.nn.dropout(in_image, keep_prob=self.keep_prob)
            # conv layer1
            conv1 = tcl.conv2d(in_image, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                               weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                               biases_initializer=tf.zeros_initializer(), activation_fn=None)
            conv1 = tf.nn.relu(conv1)
            # conv layer2
            conv2 = tcl.conv2d(conv1, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                               weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                               biases_initializer=tf.zeros_initializer(), activation_fn=None)
            conv2 = shortcut_deconv8 = tf.nn.relu(conv2)
            # conv layer3
            conv3 = tcl.conv2d(conv2, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                               weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                               biases_initializer=tf.zeros_initializer(), activation_fn=None)
            conv3 = tf.nn.relu(conv3)
            # conv layer4
            conv4 = tcl.conv2d(conv3, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                               weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                               biases_initializer=tf.zeros_initializer(), activation_fn=None)
            conv4 = shortcut_deconv6 = tf.nn.relu(conv4)
            # conv layer5
            conv5 = tcl.conv2d(conv4, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                               weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                               biases_initializer=tf.zeros_initializer(), activation_fn=None)
            conv5 = tf.nn.relu(conv5)

            """
            decoder
            """
            # deconv 6 + shortcut (residual style)
            deconv6 = tcl.conv2d_transpose(conv5, self.filter_size, self.kernel_size, self.conv_stride, padding='valid',
                                           weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                                           biases_initializer=tf.zeros_initializer(), activation_fn=None)
            deconv6 += shortcut_deconv6
            deconv6 = tf.nn.relu(deconv6)
            # deconv 7
            deconv7 = tcl.conv2d_transpose(deconv6, self.filter_size, self.kernel_size, self.conv_stride,
                                           padding='valid',
                                           weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                                           biases_initializer=tf.zeros_initializer(), activation_fn=None)
            deconv7 = tf.nn.relu(deconv7)
            # deconv 8 + shortcut
            deconv8 = tcl.conv2d_transpose(deconv7, self.filter_size, self.kernel_size, self.conv_stride,
                                           padding='valid',
                                           weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                                           biases_initializer=tf.zeros_initializer(), activation_fn=None)
            deconv8 += shortcut_deconv8
            deconv8 = tf.nn.relu(deconv8)
            # deconv 9
            deconv9 = tcl.conv2d_transpose(deconv8, self.filter_size, self.kernel_size, self.conv_stride,
                                           padding='valid',
                                           weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                                           biases_initializer=tf.zeros_initializer(), activation_fn=None)
            deconv9 = tf.nn.relu(deconv9)
            # deconv 10 + shortcut
            deconv10 = tcl.conv2d_transpose(deconv9, 1, self.kernel_size, self.conv_stride, padding='valid',
                                            weights_initializer=tf.random_normal_initializer(stddev=self.initial_std), \
                                            biases_initializer=tf.zeros_initializer(), activation_fn=None)
            if self.last_activation_fn == "sigmoid":
                output = tf.nn.sigmoid(deconv10)
            else:
                output = tf.nn.tanh(deconv10)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output
