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
        self.code_shape = [int(batch_size / 4), int(image_size[0] / 4), int(image_size[1] / 4), 4]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.ones_code = tf.ones(self.code_shape, name="ones_code")
        self.image_list = {}
        self.prob_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.G_X = Unet('G_X', ngf=ngf)
        self.D_X = Discriminator('D_X', ngf=ngf)

    def get_f(self, x, beta=0.12):
        f1 = self.norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
        f2 = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        f1 = tf.reduce_mean(f1, axis=[1, 2, 3]) - f1
        f2 = f2 - tf.reduce_mean(f2, axis=[1, 2, 3])

        f1 = self.ones * tf.cast(f1 > beta, dtype="float32")
        f2 = self.ones * tf.cast(f2 > beta, dtype="float32")

        f = f1 + f2
        f = self.ones * tf.cast(f > 0.0, dtype="float32")
        return f

    def get_mask(self, m, p=5):
        mask = 1.0 - self.ones * tf.cast(m > 0.0, dtype="float32")
        shape = m.get_shape().as_list()
        mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
        mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
        return mask

    def remove_l(self, l, f):
        l_mask = self.get_mask(l, p=0)
        f = f * l_mask
        return f

    def segmentation(self, x, EC_L, DC_L):
        l_prob = DC_L(EC_L(x))
        l_f = tf.reshape(tf.cast(tf.argmax(l_prob, axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        return l_prob, l_f

    def model(self, l_m, m, x, y, z, w):
        mask = self.get_mask(m)
        f = self.get_f(m)  # M->F
        f = self.remove_l(l_m, f)
        x_g = self.G_X(1.0-f)
        self.tenaor_name["x_g"] = str(x_g)

        j_x_g = self.D_X(x_g)
        j_x = self.D_X(x)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 35 * 2
        G_loss += self.mse_loss(x_g, x) * 2

        self.image_list["mask"] = mask
        self.image_list["f"] = f
        self.image_list["x_g"] = x_g
        self.judge_list["j_x_g"] = j_x_g
        self.judge_list["j_x"] = j_x
        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.G_X.variables
            ,
                self.D_X.variables
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
