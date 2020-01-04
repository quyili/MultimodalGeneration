# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator


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
        self.tenaor_name = {}

        self.LESP = Discriminator('LESP', ngf=ngf, output_channl=3)

    def lesion_process(self, x, LESP):
        l_r_prob = LESP(x)
        l_r = tf.reshape(tf.cast(tf.argmax(tf.reduce_mean(l_r_prob,axis=[1,2]), axis=-1), dtype=tf.float32),shape=[self.input_shape[0], 1])
        return  l_r

    def model(self, l,x):
        self.tenaor_name["l"] = str(l)
        self.tenaor_name["x"] = str(x)
        l_g_by_x = self.lesion_process(x, self.LESP)
        self.tenaor_name["l_g_by_x"] = str(l_g_by_x)
        L_loss = self.mse_loss(tf.reduce_mean(l, axis=[1, 2]), l_g_by_x) * 25
        return L_loss

    def get_variables(self):
        return self.LESP.variables

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step
        D_optimizer = make_optimizer(name='Adam_D')

        return  D_optimizer

    def loss_summary(self, L_loss):
        tf.summary.scalar('loss/L_loss', L_loss)

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
