# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
from encoder import Encoder
from VAE_encoder import VEncoder
from decoder import Decoder


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
        self.EC_F = VEncoder('EC_F', ngf=ngf)
        self.DC_F = Decoder('DC_F', ngf=ngf, output_channl=2)
        self.D_F = Discriminator('D_F', ngf=ngf)
        self.FD_F = FeatureDiscriminator('FD_F', ngf=ngf)

    def model(self, x, y, label_expand):
        # L
        l = tf.reshape(tf.cast(tf.argmax(label_expand, axis=-1), dtype=tf.float32) * 0.2,
                       shape=self.input_shape)

        # X,Y -> F
        f_x = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        f_y = self.norm(tf.reduce_max(tf.image.sobel_edges(y), axis=-1))
        f = tf.reduce_max(tf.concat([f_x, f_y], axis=-1), axis=-1, keepdims=True)
        f = f - tf.reduce_mean(f, axis=[1, 2, 3])
        f = tf.ones(self.input_shape, name="ones") * tf.cast(f > 0.07, dtype=tf.float32)

        # F -> F_R VAE
        code_f_mean, code_f_logvar = self.EC_F(f)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, dtype=tf.float32)
        code_f = code_f_mean + tf.multiply(code_f_std, code_f_epsilon)

        f_r_prob = self.DC_F(code_f)
        f_r = tf.reshape(tf.cast(tf.argmax(f_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)
        code_f_r = self.EC_F(f_r)

        # CODE_F_RM
        shape = code_f.get_shape().as_list()
        code_f_rm = tf.random_normal(shape, dtype=tf.float32)
        f_rm_prob = self.DC_F(code_f_rm)
        f_rm = tf.reshape(tf.cast(tf.argmax(f_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)
        code_f_rm_r = self.EC_F(f_rm)

        # D,FD
        j_f = self.D_F(f)
        j_f_rm = self.D_F(f_rm)

        j_code_f_rm = self.FD_F(code_f_rm)
        j_code_f = self.FD_F(code_f)

        # VAE loss
        G_loss = -50 * tf.reduce_sum(1 + code_f_logvar - tf.pow(code_f_mean, 2) - tf.exp(code_f_logvar))

        # 使得结构特征图编码服从正态分布的对抗性损失
        D_loss = self.mse_loss(j_code_f_rm, 1.0) * 10
        D_loss += self.mse_loss(j_code_f, 0.0) * 10
        G_loss += self.mse_loss(j_code_f, 1.0) * 10

        G_loss += self.mse_loss(code_f_rm, code_f_rm_r) * 5
        G_loss += self.mse_loss(code_f, code_f_r) * 5

        # 使得随机正态分布矩阵解码出结构特征图更逼真的对抗性损失
        D_loss += self.mse_loss(j_f, 1.0) * 5
        D_loss += self.mse_loss(j_f_rm, 0.0) * 5
        G_loss += self.mse_loss(j_f_rm, 1.0) * 30

        # 结构特征图两次重建融合后与原始结构特征图的两两自监督一致性损失
        G_loss += self.mse_loss(f, f_r) * 5

        f_one_hot = tf.reshape(tf.one_hot(tf.cast(f, dtype=tf.int32), depth=2, axis=-1),
                               shape=f_r_prob.get_shape().as_list())
        G_loss += self.mse_loss(f_one_hot, f_r_prob) * 5

        image_list = [x, y, l, f, f_r, f_rm]

        code_list = [code_f, code_f_r, code_f_rm, code_f_rm_r]

        j_list = [j_code_f, j_code_f_rm, j_f, j_f_rm]

        loss_list = [G_loss, D_loss]

        return image_list, code_list, j_list, loss_list, code_f_rm, code_f

    def get_variables(self):
        return [self.EC_F.variables
                + self.DC_F.variables,

                self.D_F.variables +
                self.FD_F.variables
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

    def evaluation_code(self, code_list):
        code_f, code_f_r, code_f_rm, code_f_rm_r = \
            code_list[0], code_list[1], code_list[2], code_list[3]
        list = [self.PSNR(code_f, code_f_r), self.PSNR(code_f_rm, code_f_rm_r),
                # self.SSIM(code_f, code_f_r), self.SSIM(code_f_rm, code_f_rm_r)
                ]
        return list

    def evaluation_code_summary(self, evluation_list):
        tf.summary.scalar('evaluation_code/PSNR/code_f__VS__code_f_r', evluation_list[0])
        tf.summary.scalar('evaluation_code/PSNR/code_f_rm__VS__code_f_rm_r', evluation_list[1])
        # tf.summary.scalar('evaluation_code/SSIM/code_f__VS__code_f_r', evluation_list[2])
        # tf.summary.scalar('evaluation_code/SSIM/code_f_rm__VS__code_f_rm_r', evluation_list[3])

    def evaluation(self, image_list):
        x, y, l, f, f_r, f_rm = image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5]
        list = [self.PSNR(f, f_r),
                self.SSIM(f, f_r)]
        return list

    def evaluation_summary(self, evluation_list):
        tf.summary.scalar('evaluation/PSNR/f__VS__f_r', evluation_list[0])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_r', evluation_list[1])

    def histogram_summary(self, j_list):
        j_code_f, j_code_f_rm, j_f, j_f_rm = j_list[0], j_list[1], j_list[2], j_list[3]
        tf.summary.histogram('discriminator/TRUE/j_code_f_rm', j_code_f_rm)
        tf.summary.histogram('discriminator/FALSE/j_code_f', j_code_f)
        tf.summary.histogram('discriminator/TRUE/j_f', j_f)
        tf.summary.histogram('discriminator/FALSE/j_f_rm', j_f_rm)

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def image_summary(self, image_list):
        x, y, l, f, f_r, f_rm = image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5]
        tf.summary.image('image/x', x)
        tf.summary.image('image/y', y)
        tf.summary.image('image/l_input', l)
        tf.summary.image('image/f', f)
        tf.summary.image('image/f_rm', f_rm)
        tf.summary.image('image/f_r', f_r)

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
