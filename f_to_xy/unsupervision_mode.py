# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
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
        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)
        self.D_X = Discriminator('D_X', ngf=ngf)
        self.D_Y = Discriminator('D_Y', ngf=ngf)
        self.FD_R = FeatureDiscriminator('FD_R', ngf=ngf)

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

    def get_mask(self, x):
        mask = 1.0 - self.ones * tf.cast(x > 0.02, dtype="float32")
        return mask

    def model(self, l, m, l_x, l_y, x, y):
        mask = self.get_mask(m)
        mask_x = self.get_mask(x)
        mask_y = self.get_mask(y)
        f = self.get_f(m)  # M -> F
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 5])
        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 5])
        label_expand_y = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 5])
        f_rm_expand = tf.concat([
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 0], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 1], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 2], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 3], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 4], shape=self.input_shape) + f * 0.8],
            axis=-1)

        # F_RM -> X_G,Y_G,L_G
        code_rm = self.EC_R(f_rm_expand)
        x_g = self.DC_X(code_rm)
        y_g = self.DC_Y(code_rm)
        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)

        # X_G,Y_G -> F_X_G,F_Y_G -> F_G_R
        f_x_g_r = self.get_f(x_g)
        f_y_g_r = self.get_f(y_g)

        # X_G -> L_X_G
        code_x_g = self.EC_X(x_g)
        l_g_prob_by_x = self.DC_L(code_x_g)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_G -> L_Y_G
        code_y_g = self.EC_Y(y_g)
        l_g_prob_by_y = self.DC_L(code_y_g)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # X_G -> Y_G_T
        y_g_t = self.DC_Y(code_x_g)
        # Y_G -> X_G_T
        x_g_t = self.DC_X(code_y_g)

        # X -> X_R
        code_x = self.EC_X(x)
        x_r = self.DC_X(code_x)
        # Y -> Y_R
        code_y = self.EC_Y(y)
        y_r = self.DC_Y(code_y)
        # X -> Y_T
        y_t = self.DC_Y(code_x)
        # Y -> X_T
        x_t = self.DC_X(code_y)
        # X -> L_X
        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y -> L_Y
        l_f_prob_by_y = self.DC_L(code_y)
        l_f_by_y = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_T -> X_C_R
        code_y_t = self.EC_Y(y_t)
        x_c_r = self.DC_X(code_y_t)

        # X_T -> Y_C_R
        code_x_t = self.EC_X(x_t)
        y_c_r = self.DC_Y(code_x_t)

        # D,FD
        j_x = self.D_X(x)
        j_x_g = self.D_X(x_g)
        j_x_t = self.D_X(x_t)

        j_y = self.D_Y(y)
        j_y_g = self.D_Y(y_g)
        j_y_t = self.D_Y(y_t)

        j_code_rm = self.FD_R(code_rm)
        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)

        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss = self.mse_loss(j_x, 1.0) * 30
        D_loss += self.mse_loss(j_x_g, 0.0) * 25
        D_loss += self.mse_loss(j_x_t, 0.0) * 5
        G_loss = self.mse_loss(j_x_g, 1.0) * 25
        G_loss += self.mse_loss(j_x_t, 1.0) * 5

        # 使得通过随机结构特征图生成的Y模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_y, 1.0) * 30
        D_loss += self.mse_loss(j_y_g, 0.0) * 25
        D_loss += self.mse_loss(j_y_t, 0.0) * 5
        G_loss += self.mse_loss(j_y_g, 1.0) * 25
        G_loss += self.mse_loss(j_y_t, 1.0) * 5

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_loss += self.mse_loss(j_code_rm, 0.0) * 2
        D_loss += self.mse_loss(j_code_x, 1.0)
        D_loss += self.mse_loss(j_code_y, 1.0)
        G_loss += self.mse_loss(j_code_rm, 1.0) * 2

        # 输入的结构特征图的重建自监督损失
        G_loss += self.mse_loss(f, f_x_g_r) * 20
        G_loss += self.mse_loss(f, f_y_g_r) * 20

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成X模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_x)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成Y模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_y)

        # 通过解码器生成X模态与Y模态图的编码 与 X模态与Y模态图经过编码器得到的编码 的自监督语义一致性损失
        G_loss += self.mse_loss(code_rm, code_x_g)
        G_loss += self.mse_loss(code_rm, code_y_g)
        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.5

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(y_g, y_g_t) * 2
        G_loss += self.mse_loss(x_g, x_g_t) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * mask) * 1.5
        G_loss += self.mse_loss(0.0, y_g * mask) * 1.0
        G_loss += self.mse_loss(0.0, x_g_t * mask) * 1.5
        G_loss += self.mse_loss(0.0, y_g_t * mask) * 1.0

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5
        G_loss += self.mse_loss(y, y_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_c_r) * 10
        G_loss += self.mse_loss(y, y_c_r) * 10

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) \
                  + self.mse_loss(label_expand_x[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_x, l_f_by_x) * 5

        # Y模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand_y[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_y, l_f_by_y) * 5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t) * 5
        G_loss += self.mse_loss(code_y, code_x_t) * 5

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_t * mask_y) * 1
        G_loss += self.mse_loss(0.0, y_t * mask_x) * 1
        G_loss += self.mse_loss(0.0, x_r * mask_x) * 0.1
        G_loss += self.mse_loss(0.0, y_r * mask_y) * 0.1

        self.image_list = [x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t,
                      l, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y,
                      f, f_x_g_r, f_y_g_r]

        self.code_list = [code_x, code_y, code_rm, code_x_g, code_y_g]

        self.judge_list = [j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm]

        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.EC_X.variables
                + self.EC_Y.variables
                + self.DC_X.variables
                + self.DC_Y.variables
                + self.DC_L.variables,

                self.D_X.variables
                + self.D_Y.variables
                + self.FD_R.variables
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

    def histogram_summary(self, j_list):
        j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm = \
            j_list[0], j_list[1], j_list[2], j_list[3], j_list[4], j_list[5], j_list[6]
        tf.summary.histogram('discriminator/TRUE/j_x', j_x)
        tf.summary.histogram('discriminator/TRUE/j_y', j_y)
        tf.summary.histogram('discriminator/TRUE/j_code_x', j_code_x)
        tf.summary.histogram('discriminator/TRUE/j_code_y', j_code_y)

        tf.summary.histogram('discriminator/FALSE/j_x_g', j_x_g)
        tf.summary.histogram('discriminator/FALSE/j_y_g', j_y_g)
        tf.summary.histogram('discriminator/FALSE/j_code_rm', j_code_rm)

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_list):
        code_x, code_y, code_rm, code_x_g, code_y_g = \
            code_list[0], code_list[1], code_list[2], code_list[3], code_list[4]
        list = [self.PSNR(code_x, code_y),
                self.PSNR(code_rm, code_x_g), self.PSNR(code_rm, code_y_g), self.PSNR(code_x_g, code_y_g),

                self.SSIM(code_x, code_y),
                self.SSIM(code_rm, code_x_g), self.SSIM(code_rm, code_y_g), self.SSIM(code_x_g, code_y_g)]
        return list

    def evaluation_code_summary(self, evluation_list):
        tf.summary.scalar('evaluation_code/PSNR/code_x__VS__code_y', evluation_list[0])
        tf.summary.scalar('evaluation_code/PSNR/code_rm__VS__code_x_g', evluation_list[1])
        tf.summary.scalar('evaluation_code/PSNR/code_rm__VS__code_y_g', evluation_list[2])
        tf.summary.scalar('evaluation_code/PSNR/code_x_g__VS__code_y_g', evluation_list[3])

        tf.summary.scalar('evaluation_code/SSIM/code_x__VS__code_y', evluation_list[4])
        tf.summary.scalar('evaluation_code/SSIM/code_rm__VS__code_x_g', evluation_list[5])
        tf.summary.scalar('evaluation_code/SSIM/code_rm__VS__code_y_g', evluation_list[6])
        tf.summary.scalar('evaluation_code/SSIM/code_x_g__VS__code_y_g', evluation_list[7])

    def evaluation(self, image_list):
        x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t, \
        l_input, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y, \
        f, f_x_g_r, f_y_g_r = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], image_list[
                18]
        list = [self.PSNR(x, x_t), self.PSNR(x, x_r),
                self.PSNR(y, y_t), self.PSNR(y, y_r),
                self.PSNR(x_g, x_g_t),
                self.PSNR(y_g, y_g_t),
                self.PSNR(l_input, l_f_by_x), self.PSNR(l_input, l_f_by_y),
                self.PSNR(l_input, l_g), self.PSNR(l_input, l_g_by_x), self.PSNR(l_input, l_g_by_y),
                self.PSNR(f, f_x_g_r), self.PSNR(f, f_y_g_r),

                self.SSIM(x, x_t), self.SSIM(x, x_r),
                self.SSIM(y, y_t), self.SSIM(y, y_r),
                self.SSIM(x_g, x_g_t),
                self.SSIM(y_g, y_g_t),
                self.SSIM(l_input, l_f_by_x), self.SSIM(l_input, l_f_by_y),
                self.SSIM(l_input, l_g), self.SSIM(l_input, l_g_by_x), self.SSIM(l_input, l_g_by_y),
                self.SSIM(f, f_x_g_r), self.SSIM(f, f_y_g_r),
                ]
        return list

    def evaluation_summary(self, evluation_list):
        tf.summary.scalar('evaluation/PSNR/x__VS__x_t', evluation_list[0])
        tf.summary.scalar('evaluation/PSNR/x__VS__x_r', evluation_list[1])
        tf.summary.scalar('evaluation/PSNR/y__VS__y_t', evluation_list[2])
        tf.summary.scalar('evaluation/PSNR/y__VS__y_r', evluation_list[3])
        tf.summary.scalar('evaluation/PSNR/x_g__VS__x_g_t', evluation_list[4])
        tf.summary.scalar('evaluation/PSNR/y_g__VS__y_g_t', evluation_list[5])
        tf.summary.scalar('evaluation/PSNR/l_input__VS__l_f_by_x', evluation_list[6])
        tf.summary.scalar('evaluation/PSNR/l_input__VS__l_f_by_y', evluation_list[7])
        tf.summary.scalar('evaluation/PSNR/l_input__VS__l_g', evluation_list[8])
        tf.summary.scalar('evaluation/PSNR/l_input__VS__l_g_by_x', evluation_list[9])
        tf.summary.scalar('evaluation/PSNR/l_input__VS__l_g_by_y', evluation_list[10])
        tf.summary.scalar('evaluation/PSNR/f__VS__f_x_g_r', evluation_list[11])
        tf.summary.scalar('evaluation/PSNR/f__VS__f_y_g_r', evluation_list[12])

        tf.summary.scalar('evaluation/SSIM/x__VS__x_t', evluation_list[13])
        tf.summary.scalar('evaluation/SSIM/x__VS__x_r', evluation_list[14])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_t', evluation_list[15])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_r', evluation_list[16])
        tf.summary.scalar('evaluation/SSIM/x_g__VS__x_g_t', evluation_list[17])
        tf.summary.scalar('evaluation/SSIM/y_g__VS__y_g_t', evluation_list[18])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_x', evluation_list[19])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_y', evluation_list[20])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g', evluation_list[21])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_x', evluation_list[22])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_y', evluation_list[23])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_x_g_r', evluation_list[24])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_y_g_r', evluation_list[25])

    def image_summary(self, image_list):
        x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t, \
        l_input, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y, \
        f, f_x_g_r, f_y_g_r = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], image_list[
                18]
        tf.summary.image('image/x_g', x_g)
        tf.summary.image('image/x_g_t', x_g_t)
        tf.summary.image('image/x', x)
        tf.summary.image('image/x_r', x_r)
        tf.summary.image('image/x_t', x_t)

        tf.summary.image('image/y_g', y_g)
        tf.summary.image('image/y_g_t', y_g_t)
        tf.summary.image('image/y', y)
        tf.summary.image('image/y_r', y_r)
        tf.summary.image('image/y_t', y_t)

        tf.summary.image('image/l_input', l_input)
        tf.summary.image('image/l_g', l_g)
        tf.summary.image('image/l_f_by_x', l_f_by_x)
        tf.summary.image('image/l_f_by_y', l_f_by_y)
        tf.summary.image('image/l_g_by_x', l_g_by_x)
        tf.summary.image('image/l_g_by_y', l_g_by_y)

        tf.summary.image('image/f', f)
        tf.summary.image('image/f_x_g_r', f_x_g_r)
        tf.summary.image('image/f_y_g_r', f_y_g_r)

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
