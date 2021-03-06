# _*_ coding:utf-8 _*_
import tensorflow as tf
from GAN_test_discriminator import Discriminator
from GAN_test_feature_discriminator import FeatureDiscriminator
from GAN_test_encoder import GEncoder
from GAN_test_decoder import GDecoder
from encoder import Encoder
from decoder import Decoder
import numpy as np


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

        self.EC_F = GEncoder('EC_F', ngf=ngf, units=units, keep_prob=0.85)
        self.DC_F = GDecoder('DC_F', ngf=ngf, output_channl=2, units=units)

        self.EC_M = Encoder('EC_M', ngf=ngf / 2, keep_prob=0.9)
        self.DC_M = Decoder('DC_M', ngf=ngf / 2, output_channl=2)

        self.D_F = Discriminator('D_F', ngf=ngf, keep_prob=0.85)
        self.FD_F = FeatureDiscriminator('FD_F', ngf=ngf)

    def gauss_2d_kernel(self, kernel_size=3, sigma=0.0):
        kernel = np.zeros([kernel_size, kernel_size])
        center = (kernel_size - 1) / 2
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        sum_val = 0
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
                sum_val += kernel[i, j]
        sum_val = 1 / sum_val
        return kernel * sum_val

    def gaussian_blur_op(self, image, kernel, kernel_size, cdim=3):
        # kernel as placeholder variable, so it can change
        outputs = []
        pad_w = (kernel_size * kernel_size - 1) // 2
        padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
        for channel_idx in range(cdim):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            g = tf.reshape(kernel, [1, -1, 1, 1])
            data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
            g = tf.reshape(kernel, [-1, 1, 1, 1])
            data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
            outputs.append(data_c)
        return tf.concat(outputs, axis=3)

    def gaussian_blur(self, x, sigma=0.5, alpha=0.15):
        gauss_filter = self.gauss_2d_kernel(3, sigma)
        gauss_filter = gauss_filter.astype(dtype=np.float32)
        y = self.gaussian_blur_op(x, gauss_filter, 3, cdim=1)
        y = tf.ones(y.get_shape().as_list()) * tf.cast(y > alpha, dtype="float32")
        return y

    def get_mask(self, mask, p=5):
        shape = mask.get_shape().as_list()
        mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
        mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
        return mask

    def model(self, f, mask):
        # F -> F_R VAE
        # f = self.gaussian_blur(f, sigma=0.7, alpha=0.3)
        f_one_hot = tf.reshape(tf.one_hot(tf.cast(f, dtype=tf.int32), depth=2, axis=-1),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2],
                                      2 * self.input_shape[3]])
        m_one_hot = tf.reshape(tf.one_hot(tf.cast(mask, dtype=tf.int32), depth=2, axis=-1),
                               shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2],
                                      2 * self.input_shape[3]])

        code_f_mean, code_f_logvar = self.EC_F(f / 10.0 + 0.9)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        code_f = code_f_mean + code_f_std * code_f_epsilon
        f_r_prob = self.DC_F(code_f)

        # CODE_F_RM
        code_f_rm = tf.random_normal(shape, mean=0., stddev=1., dtype=tf.float32)
        f_rm_prob = self.DC_F(code_f_rm)

        # D,FD
        j_f = self.D_F(f_one_hot)
        j_f_rm = self.D_F(f_rm_prob)

        code_f = tf.reshape(code_f, shape=[self.input_shape[0], 64, 64, -1])
        code_f_rm = tf.reshape(code_f_rm, shape=[self.input_shape[0], 64, 64, -1])
        j_code_f_rm = self.FD_F(code_f_rm)
        j_code_f = self.FD_F(code_f)

        mask_r_prob = self.DC_M(self.EC_M(f_one_hot))
        mask_rm_prob = self.DC_M(self.EC_M(f_rm_prob))

        D_loss = 0.0
        FG_loss = 0.0
        MG_loss = 0.0
        # 使得结构特征图编码服从正态分布的对抗性损失
        D_loss += self.mse_loss(j_code_f_rm, 1.0) * 10
        D_loss += self.mse_loss(j_code_f, 0.0) * 10
        FG_loss += self.mse_loss(j_code_f, 1.0) * 0.001

        FG_loss += self.mse_loss(tf.reduce_mean(code_f_mean), 0.0) * 0.001
        FG_loss += self.mse_loss(tf.reduce_mean(code_f_std), 1.0) * 0.001

        # 使得随机正态分布矩阵解码出结构特征图更逼真的对抗性损失
        D_loss += self.mse_loss(j_f, 1.0) * 25
        D_loss += self.mse_loss(j_f_rm, 0.0) * 25
        FG_loss += self.mse_loss(j_f_rm, 1.0) * 0.1

        # 结构特征图两次重建融合后与原始结构特征图的两两自监督一致性损失
        FG_loss += self.mse_loss(f_one_hot, f_r_prob) * 25

        FG_loss += tf.reduce_mean(tf.abs(f_one_hot - f_r_prob)) * 10
        FG_loss += (tf.reduce_mean(f_r_prob[:, :, :, 0]) - tf.reduce_mean(f_r_prob[:, :, :, 1])) * 0.0001
        FG_loss += (tf.reduce_mean(f_rm_prob[:, :, :, 0]) - tf.reduce_mean(f_rm_prob[:, :, :, 1])) * 0.0001

        FG_loss += self.mse_loss(0.0, m_one_hot * f_r_prob) * 1
        FG_loss += self.mse_loss(0.0, mask_rm_prob * f_rm_prob) * 1
        FG_loss += self.mse_loss(m_one_hot, mask_rm_prob) * 10

        MG_loss += self.mse_loss(m_one_hot, mask_r_prob) * 15

        new_f = tf.reshape(tf.cast(tf.argmax(f_one_hot, axis=-1), dtype=tf.float32), shape=self.input_shape)
        f_r = tf.reshape(tf.cast(tf.argmax(f_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)
        f_rm = tf.reshape(tf.cast(tf.argmax(f_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)
        mask_r = tf.reshape(tf.cast(tf.argmax(mask_r_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)
        mask_rm = tf.reshape(tf.cast(tf.argmax(mask_rm_prob, axis=-1), dtype=tf.float32), shape=self.input_shape)

        self.tenaor_name["code_f_rm"] = str(code_f_rm)
        self.tenaor_name["f_rm"] = str(f_rm)
        self.tenaor_name["j_f_rm"] = str(j_f_rm)

        image_list = [new_f, f_r, f_rm, mask, mask_r, mask_rm]
        code_list = [code_f, code_f_rm]
        j_list = [j_code_f, j_code_f_rm, j_f, j_f_rm]
        loss_list = [FG_loss + MG_loss, D_loss]

        return image_list, code_list, j_list, loss_list

    def get_variables(self):
        return [self.EC_F.variables
                + self.DC_F.variables
                + self.EC_M.variables
                + self.DC_M.variables
            ,
                self.D_F.variables +
                self.FD_F.variables
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

    def evaluation_code(self, code_list):
        code_f, code_f_rm = \
            code_list[0], code_list[1]
        list = [self.PSNR(code_f, code_f_rm)]
        return list

    def evaluation_code_summary(self, evluation_list):
        tf.summary.scalar('evaluation_code/PSNR/code_f__VS__code_f_rm', evluation_list[0])

    def evaluation(self, image_list):
        f, f_r, f_rm, mask, mask_r, mask_rm = image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], \
                                              image_list[5]
        list = [self.PSNR(f, f_r),
                self.SSIM(f, f_r),
                self.PSNR(mask, mask_r),
                self.SSIM(mask, mask_r)]
        return list

    def evaluation_summary(self, evluation_list):
        tf.summary.scalar('evaluation/PSNR/f__VS__f_r', evluation_list[0])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_r', evluation_list[1])
        tf.summary.scalar('evaluation/PSNR/mask__VS__mask_r', evluation_list[2])
        tf.summary.scalar('evaluation/SSIM/mask__VS__mask_r', evluation_list[3])

    def histogram_summary(self, j_list):
        j_code_f, j_code_f_rm, j_f, j_f_rm = j_list[0], j_list[1], j_list[2], j_list[3]
        tf.summary.histogram('discriminator/TRUE/j_code_f_rm', j_code_f_rm)
        tf.summary.histogram('discriminator/FALSE/j_code_f', j_code_f)
        tf.summary.histogram('discriminator/TRUE/j_f', j_f)
        tf.summary.histogram('discriminator/FALSE/j_f_rm', j_f_rm)

    def loss_summary(self, loss_list):
        FG_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/FG_loss', FG_loss)
        # tf.summary.scalar('loss/MG_loss', MG_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def image_summary(self, image_list):
        f, f_r, f_rm, mask, mask_r, mask_rm = image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], \
                                              image_list[5]
        tf.summary.image('image/f', f)
        tf.summary.image('image/f_rm', f_rm)
        tf.summary.image('image/f_r', f_r)
        # tf.summary.image('image/f_one_hot1', f_one_hot[:,:,:,0:1])
        # tf.summary.image('image/f_one_hot2', f_one_hot[:,:,:,1:2])
        # tf.summary.image('image/f_r_prob1', f_r_prob[:,:,:,0:1])
        # tf.summary.image('image/f_r_prob2', f_r_prob[:,:,:,1:2])
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
