# _*_ coding:utf-8 _*_
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
import logging


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
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
        self.ones = tf.ones(self.input_shape, name="ones")
        self.image_list = {}
        self.prob_list = {}
        self.code_list = {}
        self.judge_list = {}
        self.tensor_name = {}

        self.EC_L = Encoder('EC_L', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)

    def segmentation(self,x, y, z, w):
        l_prob=self.DC_L(self.EC_L(tf.concat([x, y, z, w], axis=-1)))
        l_f = tf.reshape(tf.cast(tf.argmax(l_prob, axis=-1), dtype=tf.float32) * 0.25,shape=self.input_shape)
        return l_prob,l_f

    def model(self, l, x, y, z, w):
        self.tensor_name["l"] = str(l)
        self.tensor_name["x"] = str(x)
        self.tensor_name["y"] = str(y)
        self.tensor_name["z"] = str(z)
        self.tensor_name["w"] = str(w)
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        l = l * 0.25
        l_f_prob, l_f = self.segmentation(x, y, z, w)
        self.tensor_name["l_f"] = str(l_f)

        G_loss = 0.0
        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_f_prob[:, :, :, 0]) \
                  + self.mse_loss(label_expand[:, :, :, 1],
                                  l_f_prob[:, :, :, 1]) * 15 \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_f_prob[:, :, :, 2]) * 85 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_f_prob[:, :, :, 3]) * 85 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_f_prob[:, :, :, 4]) * 85
        G_loss += self.mse_loss(l, l_f) * 25

        self.image_list["l"] = l
        self.image_list["x"] = x
        self.image_list["y"] = y
        self.image_list["z"] = z
        self.image_list["w"] = w
        self.prob_list["label_expand"] = label_expand
        self.prob_list["l_f_prob"] = l_f_prob
        self.image_list["l_f"] = l_f
        return G_loss

    def get_variables(self):
        return [self.EC_L.variables
                + self.DC_L.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

    def loss_summary(self, G_loss):
        tf.summary.scalar('loss/G_loss', G_loss)

    def evaluation(self, image_dirct):
        self.name_list_true = ["l",]
        self.name_list_false = ["l_f"]
        dice_score_list = []
        mse_list = []
        for i in range(len(self.name_list_true)):
            dice_score_list.append(
                self.dice_score(image_dirct[self.name_list_true[i]] * 4, image_dirct[self.name_list_false[i]] * 4))
            mse_list.append(
                self.mse_loss(image_dirct[self.name_list_true[i]] * 4, image_dirct[self.name_list_false[i]] * 4))
        return dice_score_list, mse_list

    def evaluation_summary(self, ssim_list):
        for i in range(len(self.name_list_true)):
            tf.summary.scalar("evaluation/" + self.name_list_true[i] + "__VS__" + self.name_list_false[i], ssim_list[i])

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

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

    def dice_score(self, output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice

    def cos_score(self, output, target, axis=(1, 2, 3), smooth=1e-5):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(output), axis))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(target), axis))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(output, target), axis)
        score = tf.reduce_mean(tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + smooth))
        return score

    def euclidean_distance(self, output, target, axis=(1, 2, 3)):
        euclidean = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(output - target), axis)))
        return euclidean

    def MSE(self, output, target):
        mse = tf.reduce_mean(tf.square(output - target))
        return mse

    def MAE(self, output, target):
        mae = tf.reduce_mean(tf.abs(output - target))
        return mae

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
