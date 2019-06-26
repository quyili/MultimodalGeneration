# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
from encoder import Encoder
from shared_decoder import SDecoder
from m_decoder import MDecoder
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
        self.input_shape = [batch_size, image_size[0], image_size[1], image_size[2]]
        self.code_shape = [batch_size, int(image_size[0] / 4), int(image_size[1] / 4), 4]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.ones_code = tf.ones(self.code_shape, name="ones_code")

        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)

        self.EC_M = Encoder('EC_M', ngf=ngf)
        self.DC_M = Decoder('DC_M', ngf=ngf)
        self.D_M = Discriminator('D_M', ngf=ngf)

        self.FD_R = FeatureDiscriminator('FD_R', ngf=ngf)

    def get_f(self, x):
        f = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        f = f - tf.reduce_mean(f, axis=[1, 2, 3])
        f = self.ones * tf.cast(f > 0.085, dtype=tf.float32)
        return f

    def select_f(self, x, y, z, w):
        rand_f = tf.random_uniform([], 0, 4, dtype=tf.int32)
        m = tf.case({tf.equal(rand_f, 0): lambda: x,
                     tf.equal(rand_f, 1): lambda: y,
                     tf.equal(rand_f, 2): lambda: z,
                     tf.equal(rand_f, 3): lambda: w}, exclusive=True)
        f = self.get_f(m)  # M -> F
        return f

    def gen(self, f, l, c1_code, c2_code, G_loss=0.0):
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=6),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        f_rm_expand = tf.concat([
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 0], shape=self.input_shape)
            + tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 1], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 2], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 3], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 4], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 5], shape=self.input_shape) + f * 0.8],
            axis=-1)
        # F_RM -> X_G,Y_G,L_G
        code_rm = self.EC_R(f_rm_expand)
        l_g_prob = self.DC_L(code_rm)

        x_g = self.DC_M(tf.concat([code_rm, c1_code], axis=-1))
        y_g = self.DC_M(tf.concat([code_rm, c2_code], axis=-1))
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)

        # X_G,Y_G -> F_X_G,F_Y_G -> F_G_R
        f_x_g_r = self.get_f(x_g)
        f_y_g_r = self.get_f(y_g)

        # X_G -> L_X_G
        code_x_g = self.EC_M(x_g)
        l_g_prob_by_x = self.DC_L(code_x_g)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_G -> L_Y_G
        code_y_g = self.EC_M(y_g)
        l_g_prob_by_y = self.DC_L(code_y_g)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # X_G -> Y_G_T
        y_g_t = self.DC_M(tf.concat([code_x_g, c2_code], axis=-1))
        # Y_G -> X_G_T
        x_g_t = self.DC_M(tf.concat([code_y_g, c1_code], axis=-1))

        # 输入的结构特征图的重建自监督损失
        G_loss += self.mse_loss(f, f_x_g_r) * 5
        G_loss += self.mse_loss(f, f_y_g_r) * 5

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成X模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_x[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g_by_x)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成Y模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_y[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g_by_y)

        # 通过解码器生成X模态与Y模态图的编码 与 X模态与Y模态图经过编码器得到的编码 的自监督语义一致性损失
        G_loss += self.mse_loss(code_rm, code_x_g)
        G_loss += self.mse_loss(code_rm, code_y_g)
        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.5

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(y_g, y_g_t) * 2
        G_loss += self.mse_loss(x_g, x_g_t) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * label_expand[0]) * 1.5
        G_loss += self.mse_loss(0.0, y_g * label_expand[0]) * 1.0
        G_loss += self.mse_loss(0.0, x_g_t * label_expand[0]) * 1.5
        G_loss += self.mse_loss(0.0, y_g_t * label_expand[0]) * 1.0

        self.image_list=[f, l, x_g, y_g, x_g_t, y_g_t, l_g, l_g_by_x, l_g_by_y, f_x_g_r, f_y_g_r]
        self.code_list=[code_rm, code_x_g, code_y_g]

        return G_loss

    def translate(self, x, y, l_x, l_y, c1_code, c2_code, G_loss=0.0):
        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=6),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        label_expand_y = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=6),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        # X -> X_R
        code_x = self.EC_M(x)
        x_r = self.DC_M(tf.concat([code_x, c1_code], axis=-1))
        # Y -> Y_R
        code_y = self.EC_M(y)
        y_r = self.DC_M(tf.concat([code_y, c2_code], axis=-1))
        # X -> Y_T
        y_t = self.DC_M(tf.concat([code_x, c2_code], axis=-1))
        # Y -> X_T
        x_t = self.DC_M(tf.concat([code_y, c1_code], axis=-1))
        # X -> L_X
        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y -> L_Y
        l_f_prob_by_y = self.DC_L(code_y)
        l_f_by_y = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_T -> X_C_R
        code_y_t = self.EC_M(y_t)
        x_c_r = self.DC_M(tf.concat([code_y_t, c1_code], axis=-1))

        # X_T -> Y_C_R
        code_x_t = self.EC_M(x_t)
        y_c_r = self.DC_M(tf.concat([code_x_t, c2_code], axis=-1))

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5
        G_loss += self.mse_loss(y, y_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_c_r) * 10
        G_loss += self.mse_loss(y, y_c_r) * 10

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) \
                  + self.mse_loss(label_expand_x[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) \
                  + self.mse_loss(label_expand_x[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l_x, l_f_by_x) * 5

        # Y模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand_y[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) \
                  + self.mse_loss(label_expand_y[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l_y, l_f_by_y) * 5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t) * 5
        G_loss += self.mse_loss(code_y, code_x_t) * 5

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_r * label_expand_x[0]) * 0.5
        G_loss += self.mse_loss(0.0, y_r * label_expand_y[0]) * 0.5
        G_loss += self.mse_loss(0.0, x_t * label_expand_y[0]) * 0.5
        G_loss += self.mse_loss(0.0, y_t * label_expand_x[0]) * 0.5

        self.image_list.extend([x, y, x_r, y_r, x_t, y_t, x_c_r, y_c_r, l_f_by_x, l_f_by_y])
        self.code_list.extend([code_x, code_y])

        return G_loss

    def judge(self, x, y, cx, cy, G_loss=0.0, D_loss=0.0):
        x_g, y_g = self.image_list[0], self.image_list[1]
        code_rm, code_x, code_y = self.code_list[0], self.code_list[3], self.code_list[4]
        j_x, j_x_c = self.D_M(x)
        j_x_g, j_x_g_c = self.D_M(x_g)
        j_y, j_y_c = self.D_M(y)
        j_y_g, j_y_g_c = self.D_M(y_g)

        j_code_rm = self.FD_R(code_rm)
        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)

        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 25
        D_loss += self.mse_loss(j_x_g, 0.0) * 25
        G_loss += self.mse_loss(j_x_g, 1.0) * 25

        # 使得通过随机结构特征图生成的Y模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_y, 1.0) * 25
        D_loss += self.mse_loss(j_y_g, 0.0) * 25
        G_loss += self.mse_loss(j_y_g, 1.0) * 25

        # TODO 交叉熵损失函数
        D_loss += self.mse_loss(j_x_c, cx) * 25
        D_loss += self.mse_loss(j_y_c, cy) * 25
        G_loss += self.mse_loss(j_x_g_c, cx) * 25
        G_loss += self.mse_loss(j_y_g_c, cy) * 25

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_loss += self.mse_loss(j_code_rm, 0.0) * 2
        D_loss += self.mse_loss(j_code_x, 1.0)
        D_loss += self.mse_loss(j_code_y, 1.0)
        G_loss += self.mse_loss(j_code_rm, 1.0) * 2

        self.judge_list=[j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm]

        return G_loss, D_loss

    def model(self, f, l, m1, m2, c1, c2):
        c1_code = self.ones_code * tf.one_hot(c1, depth=4)
        c2_code = self.ones_code * tf.one_hot(c2, depth=4)

        # 生成训练过程
        G_loss = self.gen(f, l, c1_code, c2_code, G_loss=0.0)

        # 辅助训练过程
        G_loss = self.translate(m1, m2, l, l, c1_code, c2_code, G_loss=G_loss)

        # 鉴别器训练过程
        G_loss, D_loss = self.judge(m1, m2, c1, c2, G_loss=G_loss, D_loss=0.0)

        loss_list = [G_loss, D_loss]
        return loss_list

    def get_variables(self):
        return [self.EC_M.variables
                + self.DC_M.variables
                + self.EC_R.variables
                + self.DC_L.variables
            ,
                self.D_M.variables
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

    def histogram_summary(self, j_list, m="T1_T2"):
        j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm = \
            j_list[0], j_list[1], j_list[2], j_list[3], j_list[4], j_list[5], j_list[6]
        tf.summary.histogram('discriminator/' + m + '/TRUE/j_x', j_x)
        tf.summary.histogram('discriminator/' + m + '/TRUE/j_y', j_y)
        tf.summary.histogram('discriminator/' + m + '/TRUE/j_code_x', j_code_x)
        tf.summary.histogram('discriminator/' + m + '/TRUE/j_code_y', j_code_y)

        tf.summary.histogram('discriminator/' + m + '/FALSE/j_x_g', j_x_g)
        tf.summary.histogram('discriminator/' + m + '/FALSE/j_y_g', j_y_g)
        tf.summary.histogram('discriminator/' + m + '/FALSE/j_code_rm', j_code_rm)

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_list):
        code_rm, code_x_g, code_y_g, \
        code_x, code_y = \
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
        f, l, x_g, y_g, x_g_t, y_g_t, l_g, l_g_by_x, l_g_by_y, f_x_g_r, f_y_g_r, \
        x, y, x_r, y_r, x_t, y_t, x_c_r, y_c_r, l_f_by_x, l_f_by_y = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], \
            image_list[18], image_list[19], image_list[20]

        list = [self.PSNR(x, x_t), self.PSNR(x, x_r),
                self.PSNR(y, y_t), self.PSNR(y, y_r),
                self.PSNR(x_g, x_g_t),
                self.PSNR(y_g, y_g_t),
                self.PSNR(l, l_f_by_x), self.PSNR(l, l_f_by_y),
                self.PSNR(l, l_g), self.PSNR(l, l_g_by_x), self.PSNR(l, l_g_by_y),
                self.PSNR(f, f_x_g_r), self.PSNR(f, f_y_g_r),

                self.SSIM(x, x_t), self.SSIM(x, x_r),
                self.SSIM(y, y_t), self.SSIM(y, y_r),
                self.SSIM(x_g, x_g_t),
                self.SSIM(y_g, y_g_t),
                self.SSIM(l, l_f_by_x), self.SSIM(l, l_f_by_y),
                self.SSIM(l, l_g), self.SSIM(l, l_g_by_x), self.SSIM(l, l_g_by_y),
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
        tf.summary.scalar('evaluation/PSNR/f__VS__f_xy_g_r', evluation_list[11])
        tf.summary.scalar('evaluation/PSNR/f__VS__f_xy_g_r', evluation_list[12])

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
        tf.summary.scalar('evaluation/SSIM/f__VS__f_xy_g_r', evluation_list[24])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_xy_g_r', evluation_list[25])

    def image_summary(self, image_list, m="T1_T2"):
        f, l, x_g, y_g, x_g_t, y_g_t, l_g, l_g_by_x, l_g_by_y, f_x_g_r, f_y_g_r, \
        x, y, x_r, y_r, x_t, y_t, x_c_r, y_c_r, l_f_by_x, l_f_by_y = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], \
            image_list[18], image_list[19], image_list[20]
        tf.summary.image('image/' + m + '/x_g', x_g)
        tf.summary.image('image/' + m + '/x_g_t', x_g_t)
        tf.summary.image('image/' + m + '/x', x)
        tf.summary.image('image/' + m + '/x_r', x_r)
        tf.summary.image('image/' + m + '/x_t', x_t)

        tf.summary.image('image/' + m + '/y_g', y_g)
        tf.summary.image('image/' + m + '/y_g_t', y_g_t)
        tf.summary.image('image/' + m + '/y', y)
        tf.summary.image('image/' + m + '/y_r', y_r)
        tf.summary.image('image/' + m + '/y_t', y_t)

        tf.summary.image('image/' + m + '/l_input', l)
        tf.summary.image('image/' + m + '/l_g', l_g)
        tf.summary.image('image/' + m + '/l_f_by_x', l_f_by_x)
        tf.summary.image('image/' + m + '/l_f_by_y', l_f_by_y)
        tf.summary.image('image/' + m + '/l_g_by_x', l_g_by_x)
        tf.summary.image('image/' + m + '/l_g_by_y', l_g_by_y)

        tf.summary.image('image/' + m + '/f', f)
        tf.summary.image('image/' + m + '/f_x_g_r', f_x_g_r)
        tf.summary.image('image/' + m + '/f_y_g_r', f_y_g_r)

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
