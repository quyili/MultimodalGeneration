# _*_ coding:utf-8 _*_
import tensorflow as tf
from ssd import Detector
from discriminator import Discriminator
from unet import Unet
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
        self.judge_list = {}
        self.classes_size = classes_size

        self.G_X = Unet('G_X', ngf=ngf, output_channl=3, keep_prob=0.97)
        self.D_X = Discriminator('D_X', ngf=ngf, keep_prob=0.9)
        self.LESP = Detector('LESP', ngf, classes_size=classes_size, keep_prob=0.99, input_channl=image_size[2])

    def pred(self, classes_size, feature_class, background_classes_val, all_default_boxs_len):
        feature_class_softmax = tf.nn.softmax(logits=feature_class, dim=-1)
        background_filter = np.ones(classes_size, dtype=np.float32)
        background_filter[background_classes_val] = 0
        background_filter = tf.constant(background_filter)
        feature_class_softmax = tf.multiply(feature_class_softmax, background_filter)
        feature_class_softmax = tf.reduce_max(feature_class_softmax, 2)
        box_top_set = tf.nn.top_k(feature_class_softmax, int(all_default_boxs_len / 20))
        box_top_index = box_top_set.indices
        box_top_value = box_top_set.values
        return feature_class_softmax, box_top_index, box_top_value

    def model(self, l, f, mask, x, groundtruth_class, groundtruth_location,
              groundtruth_positives, groundtruth_negatives):
        new_f = f + tf.random_uniform([self.input_shape[0], self.input_shape[1],
                                       self.input_shape[2], 1], minval=0.5, maxval=0.6,
                                      dtype=tf.float32) * (1.0 - mask) * (1.0 - f)
        f_rm_expand = tf.concat([new_f,l + 0.1],axis=-1)
        x_g = self.G_X(f_rm_expand)
        self.tenaor_name["x_g"] = str(x_g)

        j_x_g = self.D_X(x_g)
        j_x = self.D_X(x)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 2

        G_loss += self.mse_loss(x_g, x) * 5
        G_loss += self.mse_loss(x_g * mask, x * mask) * 0.001

        feature_class, feature_location = self.LESP(x_g)
        groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                               labels=groundtruth_class)
        loss_location = tf.div(tf.reduce_sum(tf.multiply(
            tf.reduce_sum(self.smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                          reduction_indices=2), groundtruth_positives), reduction_indices=1),
            tf.reduce_sum(groundtruth_positives, reduction_indices=1))
        loss_class = tf.div(
            tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
            tf.reduce_sum(groundtruth_count, reduction_indices=1))
        loss_all = tf.reduce_sum(tf.add(loss_class, loss_location))

        image_list = {}
        image_list["mask"] = mask
        image_list["f"] = f
        image_list["new_f"] = new_f
        image_list["x"] = x
        image_list["x_g"] = x_g
        self.judge_list["j_x_g"] = j_x_g
        self.judge_list["j_x"] = j_x

        loss_list = [G_loss, D_loss]
        detect_loss_list = [loss_all, loss_class, loss_location]

        return loss_list, image_list, detect_loss_list, feature_class, feature_location

    def get_variables(self):
        return [self.G_X.variables,
                self.D_X.variables,
                self.LESP.variables]

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
            tf.summary.image('discriminator/' + key, judge_dirct[key])

    def loss_summary(self, loss_list, detect_loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)
        tf.summary.scalar('loss/detect_loss', detect_loss_list[0])

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

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

    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)),
                        tf.subtract(tf.abs(x), 0.5))
