# _*_ coding:utf-8 _*_
import tensorflow as tf
from GAN_test_discriminator import Discriminator
from encoder import Encoder
from decoder import Decoder


# from swq import Unet


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 units=4096
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
        self.tenaor_name = {}
        self.keep_prob = tf.placeholder_with_default([1.0, 1.0], shape=[2])

        self.EC_F = Encoder('EC_F', ngf=ngf, keep_prob=self.keep_prob[0])
        self.DC_F = Decoder('DC_F', ngf=ngf, output_channl=2)

        self.D_F = Discriminator('D_F', ngf=ngf, keep_prob=self.keep_prob[1])

    def model(self, f):
        # F -> F_R VAE
        f_one_hot = tf.reshape(tf.one_hot(tf.cast(f, dtype=tf.int32), depth=2, axis=-1),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2],
                                      2 * self.input_shape[3]])

        code_f = self.EC_F(tf.random_normal(self.input_shape, mean=0., stddev=1., dtype=tf.float32))
        f_rm_prob = self.DC_F(code_f)

        # D,FD
        j_f = self.D_F(f_one_hot)
        j_f_rm = self.D_F(f_rm_prob)

        D_loss = 0.0
        FG_loss = 0.0
        # 使得随机正态分布矩阵解码出结构特征图更逼真的对抗性损失
        D_loss += self.mse_loss(j_f, 1.0) * 5
        D_loss += self.mse_loss(j_f_rm, 0.0) * 5
        FG_loss += self.mse_loss(j_f_rm, 1.0) * 1

        new_f = tf.reshape(tf.cast(tf.argmax(f_one_hot, axis=-1), dtype=tf.float32), shape=self.input_shape)
        f_rm = tf.reshape(tf.cast(tf.argmax(f_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        self.tenaor_name["f_rm"] = str(f_rm)
        self.tenaor_name["j_f_rm"] = str(j_f_rm)

        image_list = [new_f, f_rm, f_one_hot, f_rm_prob]
        j_list = [j_f, j_f_rm]
        loss_list = [FG_loss, D_loss]

        return image_list, j_list, loss_list

    def get_variables(self):
        return [self.EC_F.variables
                + self.DC_F.variables
            ,
                self.D_F.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        FG_optimizer = make_optimizer(name='Adam_FG')
        MG_optimizer = make_optimizer(name='Adam_MG')
        D_optimizer = make_optimizer(name='Adam_D')

        return FG_optimizer, MG_optimizer, D_optimizer

    def histogram_summary(self, j_list):
        j_f, j_f_rm = j_list[0], j_list[1]
        tf.summary.histogram('discriminator/TRUE/j_f', j_f)
        tf.summary.histogram('discriminator/FALSE/j_f_rm', j_f_rm)

    def loss_summary(self, loss_list):
        FG_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/FG_loss', FG_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def image_summary(self, image_list):
        f, f_rm, f_one_hot, f_rm_prob = image_list[0], image_list[1], image_list[2], image_list[3]
        tf.summary.image('image/f', f)
        tf.summary.image('image/f_rm', f_rm)
        tf.summary.image('image/f_one_hot1', f_one_hot[:, :, :, 0:1])
        tf.summary.image('image/f_one_hot2', f_one_hot[:, :, :, 1:2])
        tf.summary.image('image/f_rm_prob1', f_rm_prob[:, :, :, 0:1])
        tf.summary.image('image/f_rm_prob2', f_rm_prob[:, :, :, 1:2])

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
