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
        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.EC_F = VEncoder('EC_F', ngf=ngf)
        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_F = Decoder('DC_F', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)
        self.D_X = Discriminator('D_X', ngf=ngf)
        self.D_Y = Discriminator('D_Y', ngf=ngf)
        self.D_F = Discriminator('D_F', ngf=ngf)
        self.FD_R = FeatureDiscriminator('FD_R', ngf=ngf)

    def model(self, x, y, label_expand):
        # L
        l = tf.reshape(tf.cast(tf.argmax(label_expand, axis=-1), dtype=tf.float32) * 0.2,
                       shape=self.input_shape)

        # X,Y -> F
        f_x = self.norm(tf.reduce_mean(tf.image.sobel_edges(x), axis=-1))
        f_y = self.norm(tf.reduce_mean(tf.image.sobel_edges(y), axis=-1))
        f = f_x * 0.5 + f_y * 0.5

        # F -> F_R
        code_f_mean, code_f_logvar = self.EC_F(f)
        shape = code_f_logvar.get_shape().as_list()
        code_f_std = tf.exp(0.5 * code_f_logvar)
        code_f_epsilon = tf.random_normal(shape, dtype=tf.float32)
        code_f = code_f_mean + tf.multiply(code_f_std, code_f_epsilon)
        f_r = self.DC_F(code_f)
        # 将F_R与输入标签信息融合
        f_r_expand = tf.concat([
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)

        # F_R-> CODE_XY
        code_rm_xy = self.EC_R(f_r_expand)

        # CODE_XY -> X_GR
        x_gr = self.DC_X(code_rm_xy)

        # CODE_XY -> Y_GR
        y_gr = self.DC_Y(code_rm_xy)

        # X_GR,Y_GR -> F_X_R,F_Y_R -> F_XY_R
        f_x_r = self.norm(tf.reduce_mean(tf.image.sobel_edges(x_gr), axis=-1))
        f_y_r = self.norm(tf.reduce_mean(tf.image.sobel_edges(y_gr), axis=-1))
        f_xy_r = f_x_r * 0.5 + f_y_r * 0.5

        # CODE_F_RM
        code_f_rm = tf.random_normal(shape, dtype=tf.float32)

        # CODE_F_RM -> F_RM
        f_rm = self.DC_F(code_f_rm)
        # 将F_RM与输入标签信息融合
        f_rm_expand = tf.concat([
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)

        # F_RM -> X_G,Y_G,L_G
        code_rm = self.EC_R(f_rm_expand)
        x_g = self.DC_X(code_rm)
        y_g = self.DC_Y(code_rm)
        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)

        # X_G,Y_G -> F_X_G,F_Y_G -> F_G_R
        f_x_g_r = self.norm(tf.reduce_mean(tf.image.sobel_edges(x_g), axis=-1))
        f_y_g_r = self.norm(tf.reduce_mean(tf.image.sobel_edges(y_g), axis=-1))
        f_xy_g_r = f_x_g_r * 0.5 + f_y_g_r * 0.5

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

        # D,FD
        j_f = self.D_F(f)
        j_f_1 = self.D_F(f_x * 0.25 + f_y * 0.75)
        j_f_2 = self.D_F(f_x * 0.75 + f_y * 0.25)
        j_f_rm = self.D_F(f_rm)
        j_f_rm_1 = self.D_F(
            self.DC_F(tf.truncated_normal(shape, dtype=tf.float32)))
        j_f_rm_2 = self.D_F(
            self.DC_F(tf.truncated_normal(shape, dtype=tf.float32)))

        # j_code_f = self.FD_F(code_f)
        # j_code_f_rm = self.FD_F(code_f_rm)
        # # 使得结构特征图编码服从正态分布的对抗性损失
        # D_loss = self.mse_loss(j_code_f_rm, 1.0) * 30
        # D_loss += self.mse_loss(j_code_f, 0.0) * 30
        # G_loss = self.mse_loss(j_code_f, 1.0) * 30

        j_x = self.D_X(x)
        j_x_g = self.D_X(x_g)
        j_y = self.D_Y(y)
        j_y_g = self.D_Y(y_g)

        j_code_rm = self.FD_R(code_rm)
        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)

        G_loss = -50 * tf.reduce_sum(1 + code_f_logvar - tf.pow(code_f_mean, 2) - tf.exp(code_f_logvar))

        # 使得随机正态分布矩阵解码出结构特征图更逼真的对抗性损失
        D_loss = self.mse_loss(j_f, 1.0) * 80
        D_loss += self.mse_loss(j_f_rm, 0.0) * 80
        G_loss += self.mse_loss(j_f_rm, 1.0) * 80
        D_loss += self.mse_loss(j_f_1, 1.0) * 80
        D_loss += self.mse_loss(j_f_rm_1, 0.0) * 80
        G_loss += self.mse_loss(j_f_rm_1, 1.0) * 80
        D_loss += self.mse_loss(j_f_2, 1.0) * 80
        D_loss += self.mse_loss(j_f_rm_2, 0.0) * 80
        G_loss += self.mse_loss(j_f_rm_2, 1.0) * 80

        # 结构特征图两次重建融合后与原始结构特征图的两两自监督一致性损失
        G_loss += self.mse_loss(f, f_r) * 100
        G_loss += self.mse_loss(f_r, f_xy_r) * 25
        G_loss += self.mse_loss(f, f_xy_r) * 25

        # 结构特征图第二次重建未融合时与原始未融合结构特征图的自监督一致性损失
        G_loss += self.mse_loss(f_x, f_x_r) * 25
        G_loss += self.mse_loss(f_y, f_y_r) * 25

        # 使用结构特征图生成的模态图 与 提取结构特征图的模态图 之间的自监督重建一致性损失
        G_loss += self.mse_loss(x, x_gr) * 5
        G_loss += self.mse_loss(y, y_gr) * 5

        # 使用随机正态分布矩阵解码出的结构特征图重建融合后的自监督一致性损失
        G_loss += self.mse_loss(f_rm, f_xy_g_r) * 10

        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 5
        D_loss += self.mse_loss(j_x_g, 0.0) * 5
        G_loss += self.mse_loss(j_x_g, 1.0) * 5

        # 使得通过随机结构特征图生成的Y模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_y, 1.0) * 5
        D_loss += self.mse_loss(j_y_g, 0.0) * 5
        G_loss += self.mse_loss(j_y_g, 1.0) * 5

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * label_expand[0]) * 0.5
        G_loss += self.mse_loss(0.0, y_g * label_expand[0]) * 0.5

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_loss += self.mse_loss(j_code_rm, 0.0) * 10
        D_loss += self.mse_loss(j_code_x, 1.0) * 5
        D_loss += self.mse_loss(j_code_y, 1.0) * 5
        G_loss += self.mse_loss(j_code_rm, 1.0) * 10

        # 与随机结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g)

        # 与随机结构特征图融合后输入的肿瘤分割标签图在生成X模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_x[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g_by_x)

        # 与随机结构特征图融合后输入的肿瘤分割标签图在生成Y模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_y[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g_by_y)

        # 生成的X模态与Y模态图进行分割得到的分割标签图的一致性损失
        G_loss += self.mse_loss(l_g_by_x, l_g_by_y) * 0.7

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 50 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 150 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 150 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 150
        G_loss += self.mse_loss(l, l_f_by_x) * 50

        # Y模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 50 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 150 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 150 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 150
        G_loss += self.mse_loss(l, l_f_by_y) * 50

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y) * 10

        # 通过解码器生成X模态与Y模态图的编码 与 X模态与Y模态图经过编码器得到的编码 的自监督语义一致性损失
        G_loss += self.mse_loss(code_rm, code_x_g) * 0.8
        G_loss += self.mse_loss(code_rm, code_y_g) * 0.8
        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.5

        # 通过X模态的结构特征图和Y模态的结构特征图融合图编码得到的编码结果
        # 与X模态和Y模态直接编码得到的编码结果之间的语义一致性损失
        G_loss += self.mse_loss(code_x, code_rm_xy) * 20
        G_loss += self.mse_loss(code_y, code_rm_xy) * 20

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(y_g, y_g_t) * 2
        G_loss += self.mse_loss(x_g, x_g_t) * 2

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 15
        G_loss += self.mse_loss(y, y_r) * 15

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_t) * 10
        G_loss += self.mse_loss(y, y_t) * 10

        image_list = [x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t,
                      l, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y,
                      f, f_r, f_rm, f_xy_r, f_xy_g_r]

        code_list = [code_x, code_y, code_rm, code_x_g, code_y_g]

        j_list = [j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm
                  # j_code_f, j_code_f_rm
                  ]

        loss_list = [G_loss, D_loss]

        return image_list, code_list, j_list, loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.EC_X.variables
                + self.EC_Y.variables
                + self.EC_F.variables
                + self.DC_X.variables
                + self.DC_Y.variables
                + self.DC_F.variables
                + self.DC_L.variables,

                self.D_X.variables
                + self.D_Y.variables
                + self.D_F.variables
                + self.FD_R.variables
                # + self.FD_F.variables
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
        # tf.summary.histogram('discriminator/TRUE/j_code_f_rm', j_code_f_rm)

        tf.summary.histogram('discriminator/FALSE/j_x_g', j_x_g)
        tf.summary.histogram('discriminator/FALSE/j_y_g', j_y_g)
        tf.summary.histogram('discriminator/FALSE/j_code_rm', j_code_rm)
        # tf.summary.histogram('discriminator/FALSE/j_code_f', j_code_f)

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
        f, f_r, f_rm, f_xy_r, f_xy_g_r = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], \
            image_list[18], image_list[19], image_list[20]
        list = [self.PSNR(x, x_t), self.PSNR(x, x_r),
                self.PSNR(y, y_t), self.PSNR(y, y_r),
                self.PSNR(x_g, x_g_t),
                self.PSNR(y_g, y_g_t),
                self.PSNR(l_input, l_f_by_x), self.PSNR(l_input, l_f_by_y),
                self.PSNR(l_input, l_g), self.PSNR(l_input, l_g_by_x), self.PSNR(l_input, l_g_by_y),
                self.PSNR(f, f_r), self.PSNR(f_r, f_xy_r), self.PSNR(f_rm, f_xy_g_r),

                self.SSIM(x, x_t), self.SSIM(x, x_r),
                self.SSIM(y, y_t), self.SSIM(y, y_r),
                self.SSIM(x_g, x_g_t),
                self.SSIM(y_g, y_g_t),
                self.SSIM(l_input, l_f_by_x), self.SSIM(l_input, l_f_by_y),
                self.SSIM(l_input, l_g), self.SSIM(l_input, l_g_by_x), self.SSIM(l_input, l_g_by_y),
                self.SSIM(f, f_r), self.SSIM(f_r, f_xy_r), self.SSIM(f_rm, f_xy_g_r)
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
        tf.summary.scalar('evaluation/PSNR/f__VS__f_r', evluation_list[11])
        tf.summary.scalar('evaluation/PSNR/f_r__VS__f_xy_r', evluation_list[12])
        tf.summary.scalar('evaluation/PSNR/f_rm__VS__f_xy_g_r', evluation_list[13])

        tf.summary.scalar('evaluation/SSIM/x__VS__x_t', evluation_list[14])
        tf.summary.scalar('evaluation/SSIM/x__VS__x_r', evluation_list[15])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_t', evluation_list[16])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_r', evluation_list[17])
        tf.summary.scalar('evaluation/SSIM/x_g__VS__x_g_t', evluation_list[18])
        tf.summary.scalar('evaluation/SSIM/y_g__VS__y_g_t', evluation_list[19])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_x', evluation_list[20])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_y', evluation_list[21])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g', evluation_list[22])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_x', evluation_list[23])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_y', evluation_list[24])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_r', evluation_list[25])
        tf.summary.scalar('evaluation/SSIM/f_r__VS__f_xy_r', evluation_list[26])
        tf.summary.scalar('evaluation/SSIM/f_rm__VS__f_xy_g_r', evluation_list[27])

    def image_summary(self, image_list):
        x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t, \
        l_input, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y, \
        f, f_r, f_rm, f_xy_r, f_xy_g_r = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], \
            image_list[18], image_list[19], image_list[20]
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
        tf.summary.image('image/f_rm', f_rm)
        tf.summary.image('image/f_r', f_r)
        tf.summary.image('image/f_xy_r', f_xy_r)
        tf.summary.image('image/f_xy_g_r', f_xy_g_r)

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
