# _*_ coding:utf-8 _*_
import tensorflow as tf
import ops as ops
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average

class Detector:
    def __init__(self, name, is_training=True, keep_prob=1.0):
        self.name = name
        self.reuse = False
        self.keep_prob = keep_prob

        # 是否训练
        self.isTraining = is_training
        # 分类总数量
        self.classes_size = 5
        # 卷积步长
        self.conv_strides_1 = [1, 1, 1, 1]
        self.conv_strides_2 = [1, 2, 2, 1]
        self.conv_strides_3 = [1, 3, 3, 1]
        # 池化窗口
        self.pool_size = [1, 2, 2, 1]
        # 池化步长
        self.pool_strides = [1, 2, 2, 1]
        # Batch Normalization 算法的 decay 参数
        self.conv_bn_decay = 0.99999
        # Batch Normalization 算法的 variance_epsilon 参数
        self.conv_bn_epsilon = 0.00001
        # 每个特征图单元的default box数量
        self.default_box_size = [4, 6, 6, 6, 4, 4]

    # 卷积操作
    def convolution(self, input, shape, strides, name):
        with tf.variable_scope(name):
            weight = tf.get_variable(initializer=tf.truncated_normal(shape, 0, 1), dtype=tf.float32,
                                     name=name + '_weight')
            bias = tf.get_variable(initializer=tf.truncated_normal(shape[-1:], 0, 1), dtype=tf.float32,
                                   name=name + '_bias')
            result = tf.nn.conv2d(input, weight, strides, padding='SAME', name=name + '_conv')
            result = tf.nn.bias_add(result, bias)
            result = self.batch_normalization(result, name=name + '_bn')
            result = tf.nn.relu(result, name=name + '_relu')
            return result

    # fully connect操作
    def fc(self, input, out_shape, name):
        with tf.variable_scope(name + '_fc'):
            in_shape = 1
            for d in input.get_shape().as_list()[1:]:
                in_shape *= d
            weight = tf.get_variable(initializer=tf.truncated_normal([in_shape, out_shape], 0, 1), dtype=tf.float32,
                                     name=name + '_fc_weight')
            bias = tf.get_variable(initializer=tf.truncated_normal([out_shape], 0, 1), dtype=tf.float32,
                                   name=name + '_fc_bias')
            result = tf.reshape(input, [-1, in_shape])
            result = tf.nn.xw_plus_b(result, weight, bias, name=name + '_fc_do')
            return result

    # Batch Normalization算法
    def batch_normalization(self, input, name):
        with tf.variable_scope(name):
            bn_input_shape = input.get_shape()
            moving_mean = tf.get_variable(name + '_mean', bn_input_shape[-1:], initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_variance = tf.get_variable(name + '_variance', bn_input_shape[-1:],
                                              initializer=tf.ones_initializer, trainable=False)

            def mean_var_with_update():
                mean, variance = tf.nn.moments(input, list(range(len(bn_input_shape) - 1)), name=name + '_moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, self.conv_bn_decay),
                                              assign_moving_average(moving_variance, variance,
                                                                    self.conv_bn_decay)]):
                    return tf.identity(mean), tf.identity(variance)

            # mean, variance = tf.cond(tf.cast(self.isTraining, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            mean, variance = tf.cond(tf.cast(True, tf.bool), mean_var_with_update,
                                     lambda: (moving_mean, moving_variance))
            beta = tf.get_variable(name + '_beta', bn_input_shape[-1:], initializer=tf.zeros_initializer)
            gamma = tf.get_variable(name + '_gamma', bn_input_shape[-1:], initializer=tf.ones_initializer)
            return tf.nn.batch_normalization(input, mean, variance, beta, gamma, self.conv_bn_epsilon,
                                             name + '_bn_opt')

    def __call__(self, input):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            input = tf.nn.dropout(input, keep_prob=self.keep_prob)
            # vvg16卷积层 1
            self.conv_1_1 = self.convolution(input, [3, 3, 3, 32], self.conv_strides_1, 'conv_1_1')
            self.conv_1_2 = self.convolution(self.conv_1_1, [3, 3, 32, 32], self.conv_strides_1, 'conv_1_2')
            self.conv_1_2 = tf.nn.avg_pool(self.conv_1_2, self.pool_size, self.pool_strides, padding='SAME',
                                           name='pool_1_2')
            print('##   conv_1_2 shape: ' + str(self.conv_1_2.get_shape().as_list()))
            # vvg16卷积层 2
            self.conv_2_1 = self.convolution(self.conv_1_2, [3, 3, 32, 64], self.conv_strides_1, 'conv_2_1')
            self.conv_2_2 = self.convolution(self.conv_2_1, [3, 3, 64, 64], self.conv_strides_1, 'conv_2_2')
            # self.conv_2_2 = tf.nn.avg_pool(self.conv_2_2, self.pool_size, self.pool_strides, padding='SAME',   name='pool_2_2')
            print('##   conv_2_2 shape: ' + str(self.conv_2_2.get_shape().as_list()))
            # vvg16卷积层 3
            self.conv_3_1 = self.convolution(self.conv_2_2, [3, 3, 64, 128], self.conv_strides_1, 'conv_3_1')
            self.conv_3_2 = self.convolution(self.conv_3_1, [3, 3, 128, 128], self.conv_strides_1, 'conv_3_2')
            self.conv_3_3 = self.convolution(self.conv_3_2, [3, 3, 128, 128], self.conv_strides_1, 'conv_3_3')
            self.conv_3_3 = tf.nn.avg_pool(self.conv_3_3, self.pool_size, self.pool_strides, padding='SAME',
                                           name='pool_3_3')
            print('##   conv_3_3 shape: ' + str(self.conv_3_3.get_shape().as_list()))
            # vvg16卷积层 4
            self.conv_4_1 = self.convolution(self.conv_3_3, [3, 3, 128, 256], self.conv_strides_1, 'conv_4_1')
            self.conv_4_2 = self.convolution(self.conv_4_1, [3, 3, 256, 256], self.conv_strides_1, 'conv_4_2')
            self.conv_4_3 = self.convolution(self.conv_4_2, [3, 3, 256, 256], self.conv_strides_1, 'conv_4_3')
            self.conv_4_3 = tf.nn.avg_pool(self.conv_4_3, self.pool_size, self.pool_strides, padding='SAME',
                                           name='pool_4_3')
            print('##   conv_4_3 shape: ' + str(self.conv_4_3.get_shape().as_list()))
            # vvg16卷积层 5
            self.conv_5_1 = self.convolution(self.conv_4_3, [3, 3, 256, 256], self.conv_strides_1, 'conv_5_1')
            self.conv_5_2 = self.convolution(self.conv_5_1, [3, 3, 256, 256], self.conv_strides_1, 'conv_5_2')
            self.conv_5_3 = self.convolution(self.conv_5_2, [3, 3, 256, 256], self.conv_strides_1, 'conv_5_3')
            self.conv_5_3 = tf.nn.avg_pool(self.conv_5_3, self.pool_size, self.pool_strides, padding='SAME',
                                           name='pool_5_3')
            print('##   conv_5_3 shape: ' + str(self.conv_5_3.get_shape().as_list()))
            # ssd卷积层 6
            self.conv_6_1 = self.convolution(self.conv_5_3, [3, 3, 256, 512], self.conv_strides_1, 'conv_6_1')
            print('##   conv_6_1 shape: ' + str(self.conv_6_1.get_shape().as_list()))
            # ssd卷积层 7
            self.conv_7_1 = self.convolution(self.conv_6_1, [1, 1, 512, 512], self.conv_strides_1, 'conv_7_1')
            print('##   conv_7_1 shape: ' + str(self.conv_7_1.get_shape().as_list()))
            # ssd卷积层 8
            self.conv_8_1 = self.convolution(self.conv_7_1, [1, 1, 512, 128], self.conv_strides_1, 'conv_8_1')
            self.conv_8_2 = self.convolution(self.conv_8_1, [3, 3, 128, 256], self.conv_strides_2, 'conv_8_2')
            print('##   conv_8_2 shape: ' + str(self.conv_8_2.get_shape().as_list()))
            # ssd卷积层 9
            self.conv_9_1 = self.convolution(self.conv_8_2, [1, 1, 256, 64], self.conv_strides_1, 'conv_9_1')
            self.conv_9_2 = self.convolution(self.conv_9_1, [3, 3, 64, 128], self.conv_strides_2, 'conv_9_2')
            print('##   conv_9_2 shape: ' + str(self.conv_9_2.get_shape().as_list()))
            # ssd卷积层 10
            self.conv_10_1 = self.convolution(self.conv_9_2, [1, 1, 128, 64], self.conv_strides_1, 'conv_10_1')
            self.conv_10_2 = self.convolution(self.conv_10_1, [3, 3, 64, 128], self.conv_strides_2, 'conv_10_2')
            print('##   conv_10_2 shape: ' + str(self.conv_10_2.get_shape().as_list()))
            # ssd卷积层 11
            self.conv_11 = tf.nn.avg_pool(self.conv_10_2, self.pool_size, self.pool_strides, "VALID")
            print('##   conv_11 shape: ' + str(self.conv_11.get_shape().as_list()))

            # 第 1 层 特征层，来源于conv_4_3
            self.features_1 = self.convolution(self.conv_4_3,
                                               [3, 3, 256, self.default_box_size[0] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_1')
            print('##   features_1 shape: ' + str(self.features_1.get_shape().as_list()))
            # 第 2 层 特征层，来源于conv_7_1
            self.features_2 = self.convolution(self.conv_7_1,
                                               [3, 3, 512, self.default_box_size[1] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_2')
            print('##   features_2 shape: ' + str(self.features_2.get_shape().as_list()))
            # 第 3 层 特征层，来源于conv_8_2
            self.features_3 = self.convolution(self.conv_8_2,
                                               [3, 3, 256, self.default_box_size[2] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_3')
            print('##   features_3 shape: ' + str(self.features_3.get_shape().as_list()))
            # 第 4 层 特征层，来源于conv_9_2
            self.features_4 = self.convolution(self.conv_9_2,
                                               [3, 3, 128, self.default_box_size[3] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_4')
            print('##   features_4 shape: ' + str(self.features_4.get_shape().as_list()))
            # 第 5 层 特征层，来源于conv_10_2
            self.features_5 = self.convolution(self.conv_10_2,
                                               [3, 3, 128, self.default_box_size[4] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_5')
            print('##   features_5 shape: ' + str(self.features_5.get_shape().as_list()))
            # 第 6 层 特征层，来源于conv_11
            self.features_6 = self.convolution(self.conv_11,
                                               [1, 1, 128, self.default_box_size[5] * (self.classes_size + 4)],
                                               self.conv_strides_1, 'features_6')
            print('##   features_6 shape: ' + str(self.features_6.get_shape().as_list()))

            # 特征层集合
            self.feature_maps = [self.features_1, self.features_2, self.features_3, self.features_4, self.features_5,
                                 self.features_6]
            # 获取卷积后各个特征层的shape,以便生成feature和groundtruth格式一致的训练数据
            self.feature_maps_shape = [m.get_shape().as_list() for m in self.feature_maps]

            # 整理feature数据
            self.tmp_all_feature = []
            for i, fmap in zip(range(len(self.feature_maps)), self.feature_maps):
                width = self.feature_maps_shape[i][1]
                height = self.feature_maps_shape[i][2]
                # 这里reshape目的为定位和类别2方面回归作准备
                # reshape前 shape=[None, width, height, default_box*(classes+4)]
                # reshape后 shape=[None, width*height*default_box, (classes+4) ]
                self.tmp_all_feature.append(
                    tf.reshape(fmap, [-1, (width * height * self.default_box_size[i]), (self.classes_size + 4)]))
            # 合并每张图像产生的所有特征
            self.tmp_all_feature = tf.concat(self.tmp_all_feature, axis=1)
            # 这里正式拆分为定位和类别2类数据
            self.feature_class = self.tmp_all_feature[:, :, :self.classes_size]
            self.feature_location = self.tmp_all_feature[:, :, self.classes_size:]

            print('##   feature_class shape : ' + str(self.feature_class.get_shape().as_list()))
            print('##   feature_location shape : ' + str(self.feature_location.get_shape().as_list()))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return self.feature_class,self.feature_location,self.feature_maps_shape
