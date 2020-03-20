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
             input_size：list [N, H, W, C]
             batch_size: integer, batch size
             learning_rate: float, initial learning rate for Adam
             ngf: number of base gen filters in conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.tenaor_name = {}

        self.G_X = Unet('G_X', ngf=ngf, output_channl=3)
        self.D_X = Discriminator('D_X', ngf=ngf)

    def model(self, m, s, x):
        self.tenaor_name["s"] = str(s)
        self.tenaor_name["m"] = str(m)

        new_s= s + tf.random_uniform([self.input_shape[0], self.input_shape[1],
                                    self.input_shape[2], 1], minval=0.5, maxval=0.6,
                                    dtype=tf.float32) * (1.0 - m) * (1.0 - s)

        x_g = self.G_X(new_s)
        self.tenaor_name["x_g"] = str(x_g)

        j_x_g = self.D_X(x_g)
        j_x = self.D_X(x)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 5
        D_loss += self.mse_loss(j_x_g, 0.0) * 3
        G_loss += self.mse_loss(j_x_g, 1.0) * 3

        # just for pre-training
        # G_loss += self.mse_loss(x_g, x) * 5

        image_list={}
        judge_list={}
        image_list["x_g"] = x_g
        judge_list["j_x_g"] = j_x_g
        judge_list["j_x"] = j_x,

        loss_list = [G_loss, D_loss]

        return loss_list,image_list,judge_list

    def get_variables(self):
        return [self.G_X.variables ,
                self.D_X.variables ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')

        return G_optimizer, D_optimizer

    def mse_loss(self, x, y):
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def ssim_loss(self, x, y):
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
