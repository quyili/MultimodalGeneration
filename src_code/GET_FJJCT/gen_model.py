# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from encoder import Encoder
from decoder import Decoder
import numpy as np


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

        # self.LESP = Discriminator('LESP', ngf=ngf, output_channl=3)

        self.EC_R = Encoder('EC_R', ngf=ngf, keep_prob=0.98)
        self.DC_M = Decoder('DC_M', ngf=ngf, output_channl=image_size[2], keep_prob=0.98)

        self.D_M = Discriminator('D_M', ngf=ngf, keep_prob=0.95)

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

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output

    def get_f(self, x, j=0.1):
        x1 = self.norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
        x2 = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))

        x1 = tf.reduce_mean(x1, axis=[1, 2, 3]) - x1
        x2 = x2 - tf.reduce_mean(x2, axis=[1, 2, 3])

        x1 = tf.ones(x1.get_shape().as_list()) * tf.cast(x1 > j, dtype="float32")
        x2 = tf.ones(x2.get_shape().as_list()) * tf.cast(x2 > j, dtype="float32")

        x12 = x1 + x2
        x12 = tf.ones(x12.get_shape().as_list()) * tf.cast(x12 > 0.0, dtype="float32")
        return x12

    def denoise(self, y):
        y = self.gaussian_blur(y, sigma=0.8, alpha=0.3)
        y = self.get_f(y, j=0.4)
        y = self.gaussian_blur(y, sigma=0.85, alpha=0.2)
        return y

    def model(self,
              # l,
              f_org, mask, x):
        # label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=3),
        #                         shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 3])
        f_org_1 = self.denoise(f_org[:, :, :, 0:1])
        f_org_2 = self.denoise(f_org[:, :, :, 1:2])
        f_org_3 = self.denoise(f_org[:, :, :, 2:3])
        f = tf.reshape(tf.concat([f_org_1, f_org_2, f_org_3], axis=-1), shape=self.input_shape)
        new_f = f + tf.random_uniform([self.input_shape[0], self.input_shape[1],
                                       self.input_shape[2], 3], minval=0.5, maxval=0.6,
                                      dtype=tf.float32) * (1.0 - mask) * (1.0 - f)

        code_rm = self.EC_R(new_f)
        x_g = self.DC_M(code_rm)

        # l_g_prob = self.LESP(x_g)

        j_x_g = self.D_M(x_g)
        j_x = self.D_M(x)

        D_loss = 0.0
        G_loss = 0.0
        L_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 1

        G_loss += self.mse_loss(x_g, x) * 5
        G_loss += self.mse_loss(x_g * mask, x * mask) * 0.01

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        # L_loss += self.mse_loss(tf.reduce_mean(label_expand , axis=[1, 2]),
        #                                tf.reduce_mean(l_g_prob  ,axis=[1,2])) * 0.5

        # l_r = tf.argmax(tf.reduce_mean(label_expand,axis=[1,2]), axis=-1)
        # l_g = tf.argmax(tf.reduce_mean(l_g_prob  ,axis=[1,2]), axis=-1)

        # L_acc=self.acc( l_r,l_g)

        # self.tenaor_name["l"] = str(l)
        self.tenaor_name["f"] = str(f)
        self.tenaor_name["mask"] = str(mask)
        self.tenaor_name["x"] = str(x)
        self.tenaor_name["x_g"] = str(x_g)
        # self.tenaor_name["l_g"] = str(l_g)

        image_list = {}

        image_list["mask"] = mask[:, :, :, 1:2]
        image_list["f"] = f[:, :, :, 1:2]
        image_list["new_f"] = new_f[:, :, :, 1:2]
        image_list["x"] = x[:, :, :, 1:2]
        image_list["x_g"] = x_g[:, :, :, 1:2]
        self.judge_list["j_x_g"] = j_x_g
        self.judge_list["j_x"] = j_x

        loss_list = [G_loss + L_loss, D_loss,
                     # L_loss, L_acc,l_r,l_g
                     ]

        return loss_list, image_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.DC_M.variables
            , self.D_M.variables
                # ,self.LESP.variables
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

    def histogram_summary(self, judge_dirct):
        for key in judge_dirct:
            tf.summary.image('discriminator/' + key, judge_dirct[key])

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        # L_loss,L_acc =  loss_list[2],loss_list[3]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)
        # tf.summary.scalar('loss/L_loss', L_loss)
        # tf.summary.scalar('loss/L_acc', L_acc)

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
