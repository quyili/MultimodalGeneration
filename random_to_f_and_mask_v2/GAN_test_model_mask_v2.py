# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
from GAN_test_encoder import GEncoder
from GAN_test_decoder import GDecoder
from encoder import Encoder
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
        self.ones = tf.ones(self.input_shape, name="ones")
        self.tenaor_name = {}

        self.EC_MASK = Encoder('EC_MASK', ngf=ngf)
        self.DC_MASK = Decoder('DC_MASK', ngf=ngf, output_channl=2)

        self.EC_F = GEncoder('EC_F', ngf=ngf)
        self.DC_F = GDecoder('DC_F', ngf=ngf, output_channl=2)

        self.D_F = Discriminator('D_F', ngf=ngf)
        self.FD_F = FeatureDiscriminator('FD_F', ngf=ngf)

    def get_f(self, x, beta=0.07):
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
        f = f * l_mask  # 去除肿瘤轮廓影响
        return f

    def model(self, l_m, m):
        mask = self.get_mask(m)
        f = self.get_f(m)  # M->F
        f = self.remove_l(l_m, f)

        # F -> F_R VAE
        code_f_mean, code_f_logvar = self.EC_F(f)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        code_f = code_f_mean + tf.multiply(code_f_std, code_f_epsilon)

        f_r_prob = self.DC_F(code_f)
        f_r = tf.reshape(tf.cast(tf.argmax(f_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        code_f_mask = self.EC_MASK(f)
        mask_r_prob = self.DC_MASK(code_f_mask)
        mask_r = tf.reshape(tf.cast(tf.argmax(mask_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        # CODE_F_RM
        code_f_rm = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)

        f_rm_prob = self.DC_F(code_f_rm)
        f_rm = tf.reshape(tf.cast(tf.argmax(f_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        code_f_rm_mask = self.EC_MASK(f_rm)
        mask_rm_prob = self.DC_MASK(code_f_rm_mask)
        mask_rm = tf.reshape(tf.cast(tf.argmax(mask_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        self.tenaor_name["code_f_rm"] = str(code_f_rm)
        self.tenaor_name["f_rm"] = str(f_rm)
        self.tenaor_name["mask_rm"] = str(mask_rm)

        # D,FD
        j_f = self.D_F(tf.concat([f, mask], axis=-1, name="j_f"))
        j_f_rm = self.D_F(tf.concat([f_rm, mask_rm], axis=-1, name="j_f_rm"))

        code_f = tf.reshape(code_f, shape=[-1, 64, 64, 1])
        code_f_rm = tf.reshape(code_f_rm, shape=[-1, 64, 64, 1])
        j_code_f_rm = self.FD_F(code_f_rm)
        j_code_f = self.FD_F(code_f)

        D_loss = 0.0
        G_loss = 0.0
        # 使得结构特征图编码服从正态分布的对抗性损失
        D_loss += self.mse_loss(j_code_f_rm, 1.0) * 0.1
        D_loss += self.mse_loss(j_code_f, 0.0) * 0.1
        G_loss += self.mse_loss(j_code_f, 1.0) * 0.1

        G_loss += self.mse_loss(tf.reduce_mean(code_f_mean), 0.0) * 0.1
        G_loss += self.mse_loss(tf.reduce_mean(code_f_std), 1.0) * 0.1

        # 使得随机正态分布矩阵解码出结构特征图更逼真的对抗性损失
        D_loss += self.mse_loss(j_f, 1.0) * 5
        D_loss += self.mse_loss(j_f_rm, 0.0) * 5
        G_loss += self.mse_loss(j_f_rm, 1.0) * 5

        # 结构特征图两次重建融合后与原始结构特征图的两两自监督一致性损失
        G_loss += self.mse_loss(f, f_r) * 75
        G_loss += self.mse_loss(mask, mask_r) * 25

        G_loss += self.mse_loss(0.0, f_r * mask) * 5
        G_loss += self.mse_loss(0.0, f * mask_r) * 5
        G_loss += self.mse_loss(0.0, f_r * mask_r) * 5
        G_loss += self.mse_loss(0.0, f_rm * mask_rm) * 5

        f_one_hot = tf.reshape(tf.one_hot(tf.cast(f, dtype=tf.int32), depth=2, axis=-1),
                               shape=f_r_prob.get_shape().as_list()) * 5
        G_loss += self.mse_loss(f_one_hot, f_r_prob) * 75
        mask_one_hot = tf.reshape(tf.one_hot(tf.cast(mask, dtype=tf.int32), depth=2, axis=-1),
                                  shape=mask_r_prob.get_shape().as_list())
        G_loss += self.mse_loss(mask_one_hot, mask_r_prob) * 25

        image_list = [m, f, f_r, f_rm, mask, mask_r, mask_rm]

        code_list = [code_f, code_f_rm]

        j_list = [j_code_f, j_code_f_rm, j_f, j_f_rm]

        loss_list = [G_loss, D_loss]

        return image_list, code_list, j_list, loss_list

    def get_variables(self):
        return [self.EC_F.variables
                + self.DC_F.variables
                + self.EC_MASK.variables
                + self.DC_MASK.variables,

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
        code_f, code_f_rm = \
            code_list[0], code_list[1]
        list = [self.PSNR(code_f, code_f_rm)]
        return list

    def evaluation_code_summary(self, evluation_list):
        tf.summary.scalar('evaluation_code/PSNR/code_f__VS__code_f_rm', evluation_list[0])

    def evaluation(self, image_list):
        m, f, f_r, f_rm = image_list[0], image_list[1], image_list[2], image_list[3]
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
        m, f, f_r, f_rm, mask, mask_r, mask_rm = image_list[0], image_list[1], image_list[2], image_list[3], image_list[
            4], image_list[5], image_list[6]
        tf.summary.image('image/m', m)
        tf.summary.image('image/f', f)
        tf.summary.image('image/f_rm', f_rm)
        tf.summary.image('image/f_r', f_r)
        tf.summary.image('image/mask', mask)
        tf.summary.image('image/mask_rm', mask_rm)
        tf.summary.image('image/mask_r', mask_r)

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
