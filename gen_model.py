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
        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.EC_F = Encoder('EC_F', ngf=ngf)
        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_F = Decoder('DC_F', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)
        self.D_X = Discriminator('D_X', ngf=ngf)
        self.D_Y = Discriminator('D_Y', ngf=ngf)
        self.D_F = Discriminator('D_F', ngf=ngf)
        self.FD_R = FeatureDiscriminator('FD_R', ngf=ngf)
        self.FD_F = FeatureDiscriminator('FD_F', ngf=ngf)

    def model(self, x, y, rm, label_expand):
        l = tf.reshape(tf.cast(tf.argmax(label_expand, axis=-1), dtype=tf.float32) * 0.2,
                       shape=self.input_shape)
        f_x = tf.reduce_mean(tf.image.sobel_edges(x), axis=-1)
        f_y = tf.reduce_mean(tf.image.sobel_edges(y), axis=-1)
        f = f_x * 0.5 + f_y * 0.5
        # rm_expand = tf.concat([
        #     tf.reshape(rm[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
        #     tf.reshape(rm[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
        #     tf.reshape(rm[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
        #     tf.reshape(rm[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
        #     tf.reshape(rm[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)
        # x_expand = tf.concat([
        #     tf.reshape(x[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
        #     tf.reshape(x[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
        #     tf.reshape(x[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
        #     tf.reshape(x[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
        #     tf.reshape(x[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)
        # y_expand = tf.concat([
        #     tf.reshape(y[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
        #     tf.reshape(y[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
        #     tf.reshape(y[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
        #     tf.reshape(y[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
        #     tf.reshape(y[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)

        # F -> F_R
        code_f = self.EC_F(f)
        j_code_f = self.FD_F(code_f)
        f_r = self.DC_F(code_f)

        shape = code_f.get_shape().as_list()
        code_f_rm = tf.random_normal(shape, mean=0.5, stddev=0.5, dtype=tf.float32, seed=None, name=None)
        j_code_f_rm = self.FD_F(code_f_rm)
        f_rm = self.DC_F(code_f_rm)
        j_f = self.D_F(f)
        j_f_rm = self.D_F(f_rm)

        f_r_expand = tf.concat([
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
            tf.reshape(f_r[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)
        f_rm_expand = tf.concat([
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 1], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 2], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 3], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 4], shape=self.input_shape),
            tf.reshape(f_rm[:, :, :, 0] * label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)

        # R -> X_G,Y_G,L
        code_rm = self.EC_R(f_rm_expand)
        x_g = self.DC_X(code_rm)
        y_g = self.DC_Y(code_rm)
        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)
        f_x_g_r = tf.reduce_mean(tf.image.sobel_edges(x_g), axis=-1)
        f_y_g_r = tf.reduce_mean(tf.image.sobel_edges(y_g), axis=-1)

        # X_G -> L
        code_x_g = self.EC_X(x_g)
        l_g_prob_by_x = self.DC_L(code_x_g)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_G -> L
        code_y_g = self.EC_Y(y_g)
        l_g_prob_by_y = self.DC_L(code_y_g)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # X_G -> Y_G_T
        y_g_t = self.DC_Y(code_x_g)
        # Y_G -> X_G_T
        x_g_t = self.DC_X(code_y_g)

        # F_R-> CODE
        code_rm_xy = self.EC_R(f_r_expand)

        # X -> X_GR
        x_gr = self.DC_X(code_rm_xy)
        f_x_r = tf.reduce_mean(tf.image.sobel_edges(x_gr), axis=-1)

        # Y -> Y_GR
        y_gr = self.DC_Y(code_rm_xy)
        f_y_r = tf.reduce_mean(tf.image.sobel_edges(y_gr), axis=-1)

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
        # X -> L
        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y -> L
        l_f_prob_by_y = self.DC_L(code_y)
        l_f_by_y = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)

        j_x = self.D_X(x)
        j_x_g = self.D_X(x_g)
        j_y = self.D_Y(y)
        j_y_g = self.D_Y(y_g)

        j_code_rm = self.FD_R(code_rm)
        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)

        f_xy_r = f_x_r * 0.5 + f_y_r * 0.5
        f_xy_g_r = f_x_g_r * 0.5 + f_y_g_r * 0.5

        D_loss = self.mse_loss(j_code_f_rm, 1.0) * 30
        D_loss += self.mse_loss(j_code_f, 0.0) * 30
        D_loss += self.mse_loss(j_f, 1.0) * 30
        D_loss += self.mse_loss(j_f_rm, 0.0) * 30

        G_loss = self.mse_loss(j_code_f, 1.0) * 30
        G_loss += self.mse_loss(j_f_rm, 1.0) * 30
        G_loss += self.mse_loss(f, f_r) * 50
        G_loss += self.mse_loss(f_r, f_xy_r) * 25
        G_loss += self.mse_loss(f, f_xy_r) * 25
        G_loss += self.mse_loss(f_x, f_x_r) * 25
        G_loss += self.mse_loss(f_y, f_y_r) * 25
        G_loss += self.mse_loss(f_rm, f_xy_g_r) * 10

        D_loss += self.mse_loss(j_x, 1.0) * 5
        D_loss += self.mse_loss(j_x_g, 0.0) * 5
        G_loss += self.mse_loss(j_x_g, 1.0) * 5
        G_loss += self.mse_loss(j_x_g, j_x) * 10
        G_loss += self.mse_loss(0.0, x_g * label_expand[0]) * 0.5

        D_loss += self.mse_loss(j_y, 1.0) * 5
        D_loss += self.mse_loss(j_y_g, 0.0) * 5
        G_loss += self.mse_loss(j_y_g, 1.0) * 5
        G_loss += self.mse_loss(j_y_g, j_y) * 10
        G_loss += self.mse_loss(0.0, y_g * label_expand[0]) * 0.5

        D_loss += self.mse_loss(j_code_rm, 0.0) * 10
        D_loss += self.mse_loss(j_code_x, 1.0) * 5
        D_loss += self.mse_loss(j_code_y, 1.0) * 5
        G_loss += self.mse_loss(j_code_rm, 1.0) * 10

        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g)

        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_x[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g_by_x)

        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_y[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_g_by_y)

        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_f_by_x) * 15

        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 80 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l, l_f_by_y) * 15

        G_loss += self.mse_loss(code_x, code_y) * 10
        G_loss += self.mse_loss(code_rm, code_x_g) * 0.8
        G_loss += self.mse_loss(code_rm, code_y_g) * 0.8

        G_loss += self.mse_loss(l_g_by_x, l_g_by_y) * 0.7
        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.5

        G_loss += self.mse_loss(code_x, code_rm_xy) * 20
        G_loss += self.mse_loss(code_y, code_rm_xy) * 20

        G_loss += self.mse_loss(y_g, y_g_t) * 2
        G_loss += self.mse_loss(x_g, x_g_t) * 2
        G_loss += self.mse_loss(x, x_r) * 15
        G_loss += self.mse_loss(y, y_r) * 15
        G_loss += self.mse_loss(x, x_t) * 10
        G_loss += self.mse_loss(y, y_t) * 10

        image_list = [x, y, x_g, y_g, x_g_t, y_g_t, x_r, y_r, x_t, y_t,
                      l, l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y,
                      f, f_r, f_rm, f_xy_r, f_xy_g_r]

        code_list = [code_x, code_y, code_rm, code_x_g, code_y_g]

        j_list = [j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm, j_code_f, j_code_f_rm]

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
                + self.FD_F.variables
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
        j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm, j_code_f, j_code_f_rm = \
            j_list[0], j_list[1], j_list[2], j_list[3], j_list[4], j_list[5], j_list[6], j_list[7], j_list[8]
        tf.summary.histogram('discriminator/TRUE/j_x', j_x)
        tf.summary.histogram('discriminator/TRUE/j_y', j_y)
        tf.summary.histogram('discriminator/TRUE/j_code_x', j_code_x)
        tf.summary.histogram('discriminator/TRUE/j_code_y', j_code_y)
        tf.summary.histogram('discriminator/TRUE/j_code_f_rm', j_code_f_rm)

        tf.summary.histogram('discriminator/FALSE/j_x_g', j_x_g)
        tf.summary.histogram('discriminator/FALSE/j_y_g', j_y_g)
        tf.summary.histogram('discriminator/FALSE/j_code_rm', j_code_rm)
        tf.summary.histogram('discriminator/FALSE/j_code_f', j_code_f)

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
                self.SSIM(f, f_r), self.SSIM(f_r, f_xy_r), self.SSIM(f_rm, f_xy_g_r),
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
