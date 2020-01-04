# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
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
        self.image_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.LESP = Discriminator('LESP', ngf=ngf, output_channl=3)

        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.DC_M = Decoder('DC_M', ngf=ngf)

        self.D_M = Discriminator('D_M', ngf=ngf)

    def lesion_process(self, x, LESP):
        l_r_prob = LESP(x)
        l_r = tf.reshape(tf.cast(tf.argmax(tf.reduce_mean(l_r_prob,axis=[1,2]), axis=-1), dtype=tf.float32),shape=[self.input_shape[0], 1])
        return l_r_prob, l_r

    def model(self, l,f,mask,x):
        self.tenaor_name["l"] = str(l)
        self.tenaor_name["f"] = str(f)
        self.tenaor_name["mask"] = str(mask)
        self.tenaor_name["x"] = str(x)
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=3),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 3])
        f_rm_expand = tf.concat([f ,label_expand],axis=-1)

        code_rm = self.EC_R(f_rm_expand)

        x_g = self.DC_M(code_rm)
        self.tenaor_name["x_g"] = str(x_g)

        l_g_prob_by_x, l_g_by_x = self.lesion_process(x_g, self.LESP)
        self.tenaor_name["l_g_by_x"] = str(l_g_by_x)

        j_x_g = self.D_M(x_g)
        j_x = self.D_M(x)

        D_loss = 0.0
        G_loss = 0.0
        L_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 50
        D_loss += self.mse_loss(j_x_g, 0.0) * 35
        G_loss += self.mse_loss(j_x_g, 1.0) * 35

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand,l_g_prob_by_x) * 25 * 2
        L_loss += self.mse_loss(tf.reduce_mean(l,axis=[1,2]), l_g_by_x) * 25 * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * mask) * 1.5

        self.image_list["mask"] = mask
        self.image_list["f"] = f
        self.image_list["x"] = x
        self.image_list["x_g"] = x_g
        self.judge_list["j_x_g"]= j_x_g
        self.judge_list["j_x"] = j_x

        loss_list = [G_loss+L_loss, D_loss,L_loss]

        return loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.DC_M.variables
            ,
                self.D_M.variables]

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
        G_loss, D_loss,L_loss = loss_list[0], loss_list[1],loss_list[2]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)
        tf.summary.scalar('loss/L_loss', L_loss)

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/' + key, image_dirct[key])

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
