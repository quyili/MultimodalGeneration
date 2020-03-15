# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from unet import Unet


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
        self.image_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.G_X = Unet('G_X', ngf=ngf, output_channl=3, keep_prob=0.97)

        self.D_X = Discriminator('D_X', ngf=ngf, keep_prob=0.9)

    def model(self, f, mask, x):
        self.tenaor_name["f"] = str(f)
        self.tenaor_name["mask"] = str(mask)
        self.tenaor_name["x"] = str(x)

        new_f = f + tf.random_uniform([self.input_shape[0], self.input_shape[1],
                                       self.input_shape[2], 1], minval=0.5, maxval=0.6,
                                      dtype=tf.float32) * (1.0 - mask) * (1.0 - f)
        x_g = self.G_X(new_f)
        self.tenaor_name["x_g"] = str(x_g)

        j_x_g = self.D_X(x_g)
        j_x = self.D_X(x)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 2

        G_loss += self.mse_loss(x_g, x) * 5
        G_loss += self.mse_loss(x_g * mask, 0) * 0.001

        image_list = {}
        image_list["mask"] = mask
        image_list["f"] = f
        image_list["new_f"] = new_f
        image_list["x"] = x
        image_list["x_g"] = x_g
        self.judge_list["j_x_g"] = j_x_g
        self.judge_list["j_x"] = j_x

        loss_list = [G_loss, D_loss]

        return loss_list, image_list

    def get_variables(self):
        return [self.G_X.variables
            ,
                self.D_X.variables]

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

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

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
