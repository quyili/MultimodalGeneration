# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from encoder import Encoder
import numpy as np
import dwt
from ssd import Detector


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 classes_size=5,
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
        self.ones = tf.ones(self.input_shape, name="ones")
        self.image_list = {}
        self.judge_list = {}
        self.tenaor_name = {}
        self.EC = Encoder('EC', ngf=ngf)
        self.D = Discriminator('D', ngf=ngf)
        self.LESP = Detector('LESP', ngf, classes_size=classes_size, keep_prob=0.99, input_channl=image_size[2])

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

    def get_mask(self, m):
        mask1 = self.ones * tf.cast(m > 0.0, dtype="float32")
        mask2 = self.ones * tf.cast(m < 0.0, dtype="float32")
        mask = mask1 + mask2
        return mask

    def model(self, X, low, high, groundtruth_class, groundtruth_location, groundtruth_positives,
              groundtruth_negatives):
        noise = low - high
        # LOW ->L1,L2,L3,L4
        wavelet_l1 = dwt.tf_dwt(low)
        wave_l = wavelet_l1[:, :, :, 0:1]
        wavelet_l2 = dwt.tf_dwt(wave_l)
        # LOW -> noise
        noise_t = self.EC(low, wavelet_l1)
        mask = self.get_mask(noise)
        # LOW-noise ->HIGH_T
        high_t = low - noise_t
        # HIGH_T ->L1,L2,L3,L4
        wavelet_ht1 = dwt.tf_dwt(high_t)
        wave_ht = wavelet_ht1[:, :, :, 0:1]
        wavelet_ht2 = dwt.tf_dwt(wave_ht)
        # HIGH ->L1,L2,L3,L4
        wavelet_h1 = dwt.tf_dwt(high)

        j_high_t = self.D(high_t)
        j_high = self.D(high)

        ###################################################################
        wavelet_X = dwt.tf_dwt(X)
        Y = X + self.EC(X, wavelet_X)
        j_Y = self.D(Y)

        feature_class, feature_location = self.LESP(Y)
        groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                               labels=groundtruth_class)
        # 回归损失
        loss_location = tf.div(tf.reduce_sum(tf.multiply(
            tf.reduce_sum(self.smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                          reduction_indices=2), groundtruth_positives), reduction_indices=1),
            tf.reduce_sum(groundtruth_positives, reduction_indices=1))
        # 分类损失
        loss_class = tf.div(
            tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
            tf.reduce_sum(groundtruth_count, reduction_indices=1))
        loss_all = tf.reduce_sum(tf.add(loss_class, loss_location)) * 0.5
        ######################################################################

        D_loss = 0.0
        G_loss = 0.0

        # 使得通过编码结果转换的图更逼真的对抗性损失
        D_loss += self.mse_loss(j_high, 1.0) * 5
        D_loss += self.mse_loss(j_high_t, 0.0) * 5
        D_loss += self.mse_loss(j_high_t, 0.0) * 5
        G_loss += self.mse_loss(j_high_t, 1.0) * 5

        ######################################################################
        Y_D_loss = self.mse_loss(j_Y, 0.0) * 5
        Y_G_loss = self.mse_loss(j_Y, 1.0) * 5
        ######################################################################

        # 去噪图与高剂量图的有监督小波损失
        G_loss += self.mse_loss(wavelet_h1[:, :, :, 0:1], wavelet_ht1[:, :, :, 0:1]) * 50
        G_loss += self.mse_loss(wavelet_h1[:, :, :, 1:2], wavelet_ht1[:, :, :, 1:2]) * 10
        G_loss += self.mse_loss(wavelet_h1[:, :, :, 2:3], wavelet_ht1[:, :, :, 2:3]) * 10
        G_loss += self.mse_loss(wavelet_h1[:, :, :, 3:4], wavelet_ht1[:, :, :, 3:4]) * 10

        # 低剂量图与去噪图的自监督小波损失
        G_loss += self.mse_loss(wavelet_l2[:, :, :, 0:1], wavelet_ht2[:, :, :, 0:1]) * 0.5
        G_loss += self.mse_loss(high_t, high) * 55
        G_loss += self.mse_loss(noise_t, mask * noise_t) * 5
        G_loss += self.ssim_loss((noise / 2.0) + 1.0, (noise_t / 2.0) + 1.0)

        rmse2 = self.RMSE(high, high_t)
        rmse2_n = self.RMSE(high, self.norm(high_t))
        rmse2_n2 = self.RMSE(high, self.normalize(high_t))
        bec_list = [rmse2, rmse2_n, rmse2_n2]
        self.image_list["low"] = low
        self.image_list["high"] = high
        self.image_list["noise"] = noise
        self.image_list["noise_t"] = noise_t
        self.image_list["high_t"] = self.normalize(high_t)

        self.judge_list["j_high_t"] = j_high_t
        self.judge_list["j_high"] = j_high

        loss_list = [G_loss, D_loss]
        detect_loss_list = [loss_all, Y_G_loss, Y_D_loss, loss_class, loss_location]

        return loss_list, bec_list, detect_loss_list, feature_class, feature_location

    def get_variables(self):
        return [self.EC.variables,
                self.D.variables,
                self.LESP.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')

        return G_optimizer, D_optimizer

    def histogram_summary(self, judge_dirct):
        for key in judge_dirct:
            tf.summary.image('judge/' + key, judge_dirct[key])

    def loss_summary(self, loss_list, bec_list, detect_loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        rmse2, rmse2_n, rmse2_n2 = bec_list[0], bec_list[1], bec_list[2]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)
        tf.summary.scalar('loss/rmse', rmse2)
        tf.summary.scalar('loss/rmse_n', rmse2_n)
        tf.summary.scalar('loss/rmse2_n2', rmse2_n2)
        tf.summary.scalar('loss/detect_loss', detect_loss_list[0])

    def evaluation(self, image_dirct):
        self.name_list_true = ["high", "high"]
        self.name_list_false = ["low", "high_t"]
        ssim_list = []
        psnr_list = []
        for i in range(len(self.name_list_true)):
            ssim_list.append(self.SSIM(image_dirct[self.name_list_true[i]], image_dirct[self.name_list_false[i]]))
            psnr_list.append(self.PSNR(image_dirct[self.name_list_true[i]], image_dirct[self.name_list_false[i]]))

        return ssim_list, psnr_list

    def evaluation_summary(self, ssim_list, psnr_list):
        for i in range(len(self.name_list_true)):
            tf.summary.scalar("evaluation_ssim/" + self.name_list_true[i] + "__VS__" + self.name_list_false[i],
                              ssim_list[i])
            tf.summary.scalar("evaluation_psnr/" + self.name_list_true[i] + "__VS__" + self.name_list_false[i],
                              psnr_list[i])

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def mae_loss(self, x, y):
        """ supervised loss (L1 norm)
        """
        loss = tf.reduce_mean(tf.abs(x - y))
        return loss

    def wavelet_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = 0.0
        for i in range(x.get_shape()[3].value):
            loss += self.mse_loss(x[:, :, :, i:i + 1], y[:, :, :, i:i + 1])
        return loss

    def ssim_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = (1.0 - self.SSIM(x, y)) * 20
        return loss

    def psnr_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = (50.0 - self.PSNR(x, y))
        return loss

    def PSNR(self, output, target):
        psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
        return psnr

    def SSIM(self, output, target):
        ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
        return ssim

    def RMSE(self, output, target):
        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(output, target)))
        return rmse

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output

    def normalize(self, input, max_=3072, min_=-1024):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (max_ - min_)
        return output

    def wave_norm(self, input, input_shape):
        output = tf.reshape(tf.concat([self.norm(input[:, :, :, 0:1]),
                                       self.norm(input[:, :, :, 1:2]),
                                       self.norm(input[:, :, :, 2:3]),
                                       self.norm(input[:, :, :, 3:4])], axis=-1),
                            shape=input_shape)
        return output

    # smooth_L1 算法
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)),
                        tf.subtract(tf.abs(x), 0.5))
