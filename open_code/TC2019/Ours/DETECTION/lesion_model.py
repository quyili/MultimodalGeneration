# _*_ coding:utf-8 _*_
import tensorflow as tf
from ssd import Detector
import numpy as np


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 classes_size=5,
                 ngf=64,
                 ):
        """
        Args:
          input_size：list [H, W, C]
          batch_size: integer, batch size
          learning_rate: float, initial learning rate for Adam
          ngf: number of gen filters in first conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.tenaor_name = {}
        self.classes_size=classes_size

        #病灶检测器
        self.LESP = Detector('LESP', ngf,classes_size=classes_size,keep_prob=0.99)
        #鉴别器
        # self.D_LESP = Discriminator('D_LESP', ngf,keep_prob=0.99)

    def pred(self, classes_size, feature_class, background_classes_val, all_default_boxs_len):
        # softmax归一化预测结果
        feature_class_softmax = tf.nn.softmax(logits=feature_class, dim=-1)
        # 过滤background的预测值
        background_filter = np.ones(classes_size, dtype=np.float32)
        background_filter[background_classes_val] = 0
        background_filter = tf.constant(background_filter)
        feature_class_softmax = tf.multiply(feature_class_softmax, background_filter)
        # 计算每个box的最大预测值
        feature_class_softmax = tf.reduce_max(feature_class_softmax, 2)
        # 过滤冗余的预测结果
        box_top_set = tf.nn.top_k(feature_class_softmax, int(all_default_boxs_len / 20))
        box_top_index = box_top_set.indices
        box_top_value = box_top_set.values
        return feature_class_softmax, box_top_index, box_top_value

    def model(self, input, groundtruth_class, groundtruth_location, groundtruth_positives, groundtruth_negatives):
        feature_class, feature_location = self.LESP(input)

        # j_true = self.D_LESP(input,tf.one_hot(groundtruth_class,depth=self.classes_size), groundtruth_location)
        # j_pred = self.D_LESP(input, feature_class, feature_location)

        # 损失函数
        # D_adv_loss=self.mse_loss(j_true,1.0)+self.mse_loss(j_pred,0.0)
        # G_adv_loss=self.mse_loss(j_true,0.0)
        self.groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
        self.softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                                    labels=groundtruth_class)
        self.loss_location = tf.div(tf.reduce_sum(tf.multiply(
            tf.reduce_sum(self.smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                          reduction_indices=2), groundtruth_positives), reduction_indices=1),
            tf.reduce_sum(groundtruth_positives, reduction_indices=1))
        self.loss_class = tf.div(
            tf.reduce_sum(tf.multiply(self.softmax_cross_entropy, self.groundtruth_count), reduction_indices=1),
            tf.reduce_sum(self.groundtruth_count, reduction_indices=1))
        self.loss_all = tf.reduce_sum(tf.add(self.loss_class, self.loss_location)
                                      # +G_adv_loss
                                      )

        return [self.loss_all,
                # D_adv_loss
                ], \
               feature_class, feature_location

    def get_variables(self):
        return [self.LESP.variables,
                # self.D_LESP.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        # D_optimizer = make_optimizer(name='Adam_D')

        # return [G_optimizer,D_optimizer]
        return G_optimizer

    def loss_summary(self, L_loss):
        tf.summary.scalar('loss/L_loss', L_loss[0])
        # tf.summary.scalar('loss/D_loss', L_loss[1])

    def acc(self, x, y):
        correct_prediction = tf.equal(x, y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def ssim_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = (1.0 - self.SSIM(x, y)) * 20
        return loss

    def PSNR(self, output, target):
        psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
        return psnr

    def SSIM(self, output, target):
        ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
        return ssim

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output

    # smooth_L1 算法
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)),
                        tf.subtract(tf.abs(x), 0.5))
