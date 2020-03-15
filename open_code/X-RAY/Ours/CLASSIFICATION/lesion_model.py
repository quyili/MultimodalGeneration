# _*_ coding:utf-8 _*_
import tensorflow as tf
from vgg11 import VGG


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

        self.LESP = VGG('LESP', ngf=ngf, output_channl=3, keep_prob=0.55)

    def model(self, l, x):
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=3),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 3])

        l_g_prob = self.LESP(x)

        L_loss = self.mse_loss(tf.reduce_mean(label_expand, axis=[1, 2]),
                               tf.reduce_mean(l_g_prob, axis=[1, 2]))

        l_r = tf.argmax(tf.reduce_mean(label_expand, axis=[1, 2]), axis=-1)
        l_g = tf.argmax(tf.reduce_mean(l_g_prob, axis=[1, 2]), axis=-1)

        self.tenaor_name["l"] = str(l)
        self.tenaor_name["x"] = str(x)
        self.tenaor_name["l_g"] = str(l_g)

        L_acc = self.acc(l_r, l_g)

        return [L_loss, L_acc, l_r, l_g]

    def get_variables(self):
        return self.LESP.variables

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        D_optimizer = make_optimizer(name='Adam_D')

        return D_optimizer

    def loss_summary(self, L_loss):
        tf.summary.scalar('loss/L_loss', L_loss[0])
        tf.summary.scalar('loss/L_acc', L_loss[1])

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
