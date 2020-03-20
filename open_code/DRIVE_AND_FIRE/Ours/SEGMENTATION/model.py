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

        self.G_L_X = Unet('G_L_X', ngf=ngf, output_channl=2)

    def model(self, l, x):
        l_onehot = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 5])

        l_g_prob_by_x = self.G_L_X(x)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32), shape=self.input_shape)

        G_loss = 0.0

        G_loss += self.mse_loss(l_onehot[:, :, :, 0],
                                l_g_prob_by_x[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 1],
                                  l_g_prob_by_x[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 2],
                                  l_g_prob_by_x[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 3],
                                  l_g_prob_by_x[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(l_onehot[:, :, :, 4],
                                  l_g_prob_by_x[:, :, :, 4]) * 25 * 5

        image_list={}
        image_list["l_g_by_x"] = l_g_by_x

        loss_list = [G_loss]

        return loss_list,image_list

    def get_variables(self):
        return [self.G_L_X.variables]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

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
