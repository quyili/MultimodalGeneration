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
        self.code_shape = [int(batch_size / 4), int(image_size[0] / 4), int(image_size[1] / 4), 4]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.ones_code = tf.ones(self.code_shape, name="ones_code")
        self.image_list = {}
        self.prob_list = {}
        self.code_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)

        self.EC_M = Encoder('EC_M', ngf=ngf)
        self.DC_M = Decoder('DC_M', ngf=ngf)

        self.D_M = Discriminator('D_M', ngf=ngf)
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

    # TODO input f
    def model(self, l, l_m, m, l_x, l_y, l_z, l_w, x, y, z, w):
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx,dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy,dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz,dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw,dtype=tf.int32), depth=4)

        mask = self.get_mask(m)
        f = self.get_f(m)  # M->F
        f = self.remove_l(l_m, f)
        self.tenaor_name["l"] = str(l)
        self.tenaor_name["f"] = str(f)
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                                    shape=[self.input_shape[0], self.input_shape[1],
                                                           self.input_shape[2], 5])
        l = l * 0.25
        f_rm_expand = tf.concat([
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 0],
                       shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 1],
                       shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 2],
                       shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 3],
                       shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 4],
                       shape=self.input_shape) + f * 0.8],
            axis=-1)

        code_rm = self.EC_R(f_rm_expand)

        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(
            tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        x_g = self.DC_M(tf.concat([code_rm, cx_code], axis=-1))
        y_g = self.DC_M(tf.concat([code_rm, cy_code], axis=-1))
        z_g = self.DC_M(tf.concat([code_rm, cz_code], axis=-1))
        w_g = self.DC_M(tf.concat([code_rm, cw_code], axis=-1))
        self.tenaor_name["x_g"] = str(x_g)
        self.tenaor_name["y_g"] = str(y_g)
        self.tenaor_name["z_g"] = str(z_g)
        self.tenaor_name["w_g"] = str(w_g)

        code_x_g = self.EC_M(x_g)
        code_y_g = self.EC_M(y_g)
        code_z_g = self.EC_M(z_g)
        code_w_g = self.EC_M(w_g)

        l_g_prob_by_x = self.DC_L(code_x_g)
        l_g_prob_by_y = self.DC_L(code_y_g)
        l_g_prob_by_z = self.DC_L(code_z_g)
        l_g_prob_by_w = self.DC_L(code_w_g)
        l_g_by_x = tf.reshape(
            tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_g_by_y = tf.reshape(
            tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_g_by_z = tf.reshape(
            tf.cast(tf.argmax(l_g_prob_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_g_by_w = tf.reshape(
            tf.cast(tf.argmax(l_g_prob_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        f_x_g_r = self.get_f(x_g)
        f_y_g_r = self.get_f(y_g)
        f_z_g_r = self.get_f(z_g)
        f_w_g_r = self.get_f(w_g)

        y_g_t_by_x = self.DC_M(tf.concat([code_x_g, cy_code], axis=-1))
        z_g_t_by_x = self.DC_M(tf.concat([code_x_g, cz_code], axis=-1))
        w_g_t_by_x = self.DC_M(tf.concat([code_x_g, cw_code], axis=-1))

        x_g_t_by_y = self.DC_M(tf.concat([code_y_g, cx_code], axis=-1))
        z_g_t_by_y = self.DC_M(tf.concat([code_y_g, cz_code], axis=-1))
        w_g_t_by_y = self.DC_M(tf.concat([code_y_g, cw_code], axis=-1))

        x_g_t_by_z = self.DC_M(tf.concat([code_z_g, cx_code], axis=-1))
        y_g_t_by_z = self.DC_M(tf.concat([code_z_g, cy_code], axis=-1))
        w_g_t_by_z = self.DC_M(tf.concat([code_z_g, cw_code], axis=-1))

        x_g_t_by_w = self.DC_M(tf.concat([code_w_g, cx_code], axis=-1))
        y_g_t_by_w = self.DC_M(tf.concat([code_w_g, cy_code], axis=-1))
        z_g_t_by_w = self.DC_M(tf.concat([code_w_g, cz_code], axis=-1))

        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        label_expand_y = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        label_expand_z = tf.reshape(tf.one_hot(tf.cast(l_z, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        label_expand_w = tf.reshape(tf.one_hot(tf.cast(l_w, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])

        mask_x = self.get_mask(x)
        mask_y = self.get_mask(y)
        mask_z = self.get_mask(z)
        mask_w = self.get_mask(w)

        code_x = self.EC_M(x)
        code_y = self.EC_M(y)
        code_z = self.EC_M(z)
        code_w = self.EC_M(w)

        l_f_prob_by_x = self.DC_L(code_x)
        l_f_prob_by_y = self.DC_L(code_y)
        l_f_prob_by_z = self.DC_L(code_z)
        l_f_prob_by_w = self.DC_L(code_w)
        l_f_by_x = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_f_by_y = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_f_by_z = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        l_f_by_w = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        x_r = self.DC_M(tf.concat([code_x, cx_code], axis=-1))
        y_r = self.DC_M(tf.concat([code_y, cy_code], axis=-1))
        z_r = self.DC_M(tf.concat([code_z, cz_code], axis=-1))
        w_r = self.DC_M(tf.concat([code_w, cw_code], axis=-1))

        y_t_by_x = self.DC_M(tf.concat([code_x, cy_code], axis=-1))
        code_y_t_by_x = self.EC_M(y_t_by_x)
        x_r_c_by_y = self.DC_M(tf.concat([code_y_t_by_x, cx_code], axis=-1))
        z_t_by_x = self.DC_M(tf.concat([code_x, cz_code], axis=-1))
        code_z_t_by_x = self.EC_M(z_t_by_x)
        x_r_c_by_z = self.DC_M(tf.concat([code_z_t_by_x, cx_code], axis=-1))
        w_t_by_x = self.DC_M(tf.concat([code_x, cw_code], axis=-1))
        code_w_t_by_x = self.EC_M(w_t_by_x)
        x_r_c_by_w = self.DC_M(tf.concat([code_w_t_by_x, cx_code], axis=-1))

        x_t_by_y = self.DC_M(tf.concat([code_y, cx_code], axis=-1))
        code_x_t_by_y = self.EC_M(x_t_by_y)
        y_r_c_by_x = self.DC_M(tf.concat([code_x_t_by_y, cy_code], axis=-1))
        z_t_by_y = self.DC_M(tf.concat([code_y, cz_code], axis=-1))
        code_z_t_by_y = self.EC_M(x_t_by_y)
        y_r_c_by_z = self.DC_M(tf.concat([code_z_t_by_y, cy_code], axis=-1))
        w_t_by_y = self.DC_M(tf.concat([code_y, cw_code], axis=-1))
        code_w_t_by_y = self.EC_M(x_t_by_y)
        y_r_c_by_w = self.DC_M(tf.concat([code_w_t_by_y, cy_code], axis=-1))

        x_t_by_z = self.DC_M(tf.concat([code_z, cx_code], axis=-1))
        code_x_t_by_z = self.EC_M(x_t_by_z)
        z_r_c_by_x = self.DC_M(tf.concat([code_x_t_by_z, cz_code], axis=-1))
        y_t_by_z = self.DC_M(tf.concat([code_z, cy_code], axis=-1))
        code_y_t_by_z = self.EC_M(y_t_by_z)
        z_r_c_by_y = self.DC_M(tf.concat([code_y_t_by_z, cz_code], axis=-1))
        w_t_by_z = self.DC_M(tf.concat([code_z, cw_code], axis=-1))
        code_w_t_by_z = self.EC_M(w_t_by_z)
        z_r_c_by_w = self.DC_M(tf.concat([code_w_t_by_z, cz_code], axis=-1))

        x_t_by_w = self.DC_M(tf.concat([code_w, cx_code], axis=-1))
        code_x_t_by_w = self.EC_M(x_t_by_w)
        w_r_c_by_x = self.DC_M(tf.concat([code_x_t_by_w, cw_code], axis=-1))
        y_t_by_w = self.DC_M(tf.concat([code_w, cy_code], axis=-1))
        code_y_t_by_w = self.EC_M(y_t_by_w)
        w_r_c_by_y = self.DC_M(tf.concat([code_y_t_by_w, cw_code], axis=-1))
        z_t_by_w = self.DC_M(tf.concat([code_w, cz_code], axis=-1))
        code_z_t_by_w = self.EC_M(z_t_by_w)
        w_r_c_by_z = self.DC_M(tf.concat([code_z_t_by_w, cw_code], axis=-1))

        j_x_g, j_x_g_c = self.D_M(x_g)
        j_y_g, j_y_g_c = self.D_M(y_g)
        j_z_g, j_z_g_c = self.D_M(z_g)
        j_w_g, j_w_g_c = self.D_M(w_g)

        j_x, j_x_c = self.D_M(x)
        j_y, j_y_c = self.D_M(y)
        j_z, j_z_c = self.D_M(z)
        j_w, j_w_c = self.D_M(w)

        j_x_t_by_y, j_x_t_c_by_y = self.D_M(x_t_by_y)
        j_x_t_by_z, j_x_t_c_by_z = self.D_M(x_t_by_z)
        j_x_t_by_w, j_x_t_c_by_w = self.D_M(x_t_by_w)

        j_y_t_by_x, j_y_t_c_by_x = self.D_M(y_t_by_x)
        j_y_t_by_z, j_y_t_c_by_z = self.D_M(y_t_by_z)
        j_y_t_by_w, j_y_t_c_by_w = self.D_M(y_t_by_w)

        j_z_t_by_x, j_z_t_c_by_x = self.D_M(z_t_by_x)
        j_z_t_by_y, j_z_t_c_by_y = self.D_M(z_t_by_y)
        j_z_t_by_w, j_z_t_c_by_w = self.D_M(z_t_by_w)

        j_w_t_by_x, j_w_t_c_by_x = self.D_M(w_t_by_x)
        j_w_t_by_y, j_w_t_c_by_y = self.D_M(w_t_by_y)
        j_w_t_by_z, j_w_t_c_by_z = self.D_M(w_t_by_z)

        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)
        j_code_z = self.FD_R(code_z)
        j_code_w = self.FD_R(code_w)
        j_code_rm = self.FD_R(code_rm)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 50
        D_loss += self.mse_loss(j_x_g, 0.0) * 35
        G_loss += self.mse_loss(j_x_g, 1.0) * 35
        D_loss += self.mse_loss(j_x_t_by_y, 0.0) * 5
        D_loss += self.mse_loss(j_x_t_by_z, 0.0) * 5
        D_loss += self.mse_loss(j_x_t_by_w, 0.0) * 5
        G_loss += self.mse_loss(j_x_t_by_y, 1.0) * 5
        G_loss += self.mse_loss(j_x_t_by_z, 1.0) * 5
        G_loss += self.mse_loss(j_x_t_by_w, 1.0) * 5

        D_loss += self.mse_loss(j_y, 1.0) * 50
        D_loss += self.mse_loss(j_y_g, 0.0) * 35
        G_loss += self.mse_loss(j_y_g, 1.0) * 35
        D_loss += self.mse_loss(j_y_t_by_x, 0.0) * 5
        D_loss += self.mse_loss(j_y_t_by_z, 0.0) * 5
        D_loss += self.mse_loss(j_y_t_by_w, 0.0) * 5
        G_loss += self.mse_loss(j_y_t_by_x, 1.0) * 5
        G_loss += self.mse_loss(j_y_t_by_z, 1.0) * 5
        G_loss += self.mse_loss(j_y_t_by_w, 1.0) * 5

        D_loss += self.mse_loss(j_z, 1.0) * 50
        D_loss += self.mse_loss(j_z_g, 0.0) * 35
        G_loss += self.mse_loss(j_z_g, 1.0) * 35
        D_loss += self.mse_loss(j_z_t_by_x, 0.0) * 5
        D_loss += self.mse_loss(j_z_t_by_y, 0.0) * 5
        D_loss += self.mse_loss(j_z_t_by_w, 0.0) * 5
        G_loss += self.mse_loss(j_z_t_by_x, 1.0) * 5
        G_loss += self.mse_loss(j_z_t_by_y, 1.0) * 5
        G_loss += self.mse_loss(j_z_t_by_w, 1.0) * 5

        D_loss += self.mse_loss(j_w, 1.0) * 50
        D_loss += self.mse_loss(j_w_g, 0.0) * 35
        G_loss += self.mse_loss(j_w_g, 1.0) * 35
        D_loss += self.mse_loss(j_w_t_by_x, 0.0) * 5
        D_loss += self.mse_loss(j_w_t_by_y, 0.0) * 5
        D_loss += self.mse_loss(j_w_t_by_z, 0.0) * 5
        G_loss += self.mse_loss(j_w_t_by_x, 1.0) * 5
        G_loss += self.mse_loss(j_w_t_by_y, 1.0) * 5
        G_loss += self.mse_loss(j_w_t_by_z, 1.0) * 5

        D_loss += self.mse_loss(j_x_c, cx) * 50
        D_loss += self.mse_loss(j_y_c, cy) * 50
        D_loss += self.mse_loss(j_z_c, cz) * 50
        D_loss += self.mse_loss(j_w_c, cw) * 50

        G_loss += self.mse_loss(j_x_g_c, cx) * 50
        G_loss += self.mse_loss(j_y_g_c, cy) * 50
        G_loss += self.mse_loss(j_z_g_c, cz) * 50
        G_loss += self.mse_loss(j_w_g_c, cw) * 50

        G_loss += self.mse_loss(j_x_t_c_by_y, cx) * 50
        G_loss += self.mse_loss(j_x_t_c_by_z, cx) * 50
        G_loss += self.mse_loss(j_x_t_c_by_w, cx) * 50

        G_loss += self.mse_loss(j_y_t_c_by_x, cy) * 50
        G_loss += self.mse_loss(j_y_t_c_by_z, cy) * 50
        G_loss += self.mse_loss(j_y_t_c_by_w, cy) * 50

        G_loss += self.mse_loss(j_z_t_c_by_x, cz) * 50
        G_loss += self.mse_loss(j_z_t_c_by_y, cz) * 50
        G_loss += self.mse_loss(j_z_t_c_by_w, cz) * 50

        G_loss += self.mse_loss(j_w_t_c_by_x, cw) * 50
        G_loss += self.mse_loss(j_w_t_c_by_y, cw) * 50
        G_loss += self.mse_loss(j_w_t_c_by_z, cw) * 50

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_loss += self.mse_loss(j_code_rm, 0.0) * 4
        D_loss += self.mse_loss(j_code_x, 1.0)
        D_loss += self.mse_loss(j_code_y, 1.0)
        D_loss += self.mse_loss(j_code_z, 1.0)
        D_loss += self.mse_loss(j_code_w, 1.0)
        G_loss += self.mse_loss(j_code_rm, 1.0) * 4

        # 输入的结构特征图的重建自监督损失
        G_loss += self.mse_loss(self.remove_l(l, f), self.remove_l(l, f_x_g_r)) * 10
        G_loss += self.mse_loss(self.remove_l(l, f), self.remove_l(l, f_y_g_r)) * 10
        G_loss += self.mse_loss(self.remove_l(l, f), self.remove_l(l, f_z_g_r)) * 10
        G_loss += self.mse_loss(self.remove_l(l, f), self.remove_l(l, f_w_g_r)) * 10
        G_loss += self.mse_loss(f_x_g_r, f_y_g_r) * 2
        G_loss += self.mse_loss(f_x_g_r, f_z_g_r) * 2
        G_loss += self.mse_loss(f_x_g_r, f_w_g_r) * 2
        G_loss += self.mse_loss(f_y_g_r, f_z_g_r) * 2
        G_loss += self.mse_loss(f_y_g_r, f_w_g_r) * 2
        G_loss += self.mse_loss(f_z_g_r, f_w_g_r) * 2

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_g_prob[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_g_prob[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_g_prob[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_g_prob[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g)

        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_g_prob_by_x[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1],
                                  l_g_prob_by_x[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_g_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_g_prob_by_x[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_g_prob_by_x[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_x)

        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_g_prob_by_y[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1],
                                  l_g_prob_by_y[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_g_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_g_prob_by_y[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_g_prob_by_y[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_y)

        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_g_prob_by_z[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1],
                                  l_g_prob_by_z[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_g_prob_by_z[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_g_prob_by_z[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_g_prob_by_z[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_z)

        G_loss += self.mse_loss(label_expand[:, :, :, 0],
                                l_g_prob_by_w[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1],
                                  l_g_prob_by_w[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2],
                                  l_g_prob_by_w[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3],
                                  l_g_prob_by_w[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4],
                                  l_g_prob_by_w[:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, l_g_by_w)

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_x[:, :, :, 0],
                                l_f_prob_by_x[:, :, :, 0]) \
                  + self.mse_loss(label_expand_x[:, :, :, 1],
                                  l_f_prob_by_x[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 2],
                                  l_f_prob_by_x[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 3],
                                  l_f_prob_by_x[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 4],
                                  l_f_prob_by_x[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_x, l_f_by_x) * 5

        G_loss += self.mse_loss(label_expand_y[:, :, :, 0],
                                l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand_y[:, :, :, 1],
                                  l_f_prob_by_y[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 2],
                                  l_f_prob_by_y[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 3],
                                  l_f_prob_by_y[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 4],
                                  l_f_prob_by_y[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_y, l_f_by_y) * 5

        G_loss += self.mse_loss(label_expand_z[:, :, :, 0],
                                l_f_prob_by_z[:, :, :, 0]) \
                  + self.mse_loss(label_expand_z[:, :, :, 1],
                                  l_f_prob_by_z[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_z[:, :, :, 2],
                                  l_f_prob_by_z[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_z[:, :, :, 3],
                                  l_f_prob_by_z[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_z[:, :, :, 4],
                                  l_f_prob_by_z[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_z, l_f_by_z) * 5

        G_loss += self.mse_loss(label_expand_w[:, :, :, 0],
                                l_f_prob_by_w[:, :, :, 0]) \
                  + self.mse_loss(label_expand_w[:, :, :, 1],
                                  l_f_prob_by_w[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_w[:, :, :, 2],
                                  l_f_prob_by_w[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_w[:, :, :, 3],
                                  l_f_prob_by_w[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_w[:, :, :, 4],
                                  l_f_prob_by_w[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_w, l_f_by_w) * 5

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(x_g, x_g_t_by_y) * 2  # + self.ssim_loss(x_g, x_g_t_by_y) * 2
        G_loss += self.mse_loss(x_g, x_g_t_by_z) * 2  # + self.ssim_loss(x_g, x_g_t_by_z) * 2
        G_loss += self.mse_loss(x_g, x_g_t_by_w) * 2  # + self.ssim_loss(x_g, x_g_t_by_w) * 2
        G_loss += self.mse_loss(x_g_t_by_y, x_g_t_by_z)  # + self.ssim_loss(x_g_t_by_y, x_g_t_by_z) * 2
        G_loss += self.mse_loss(x_g_t_by_y, x_g_t_by_w)  # + self.ssim_loss(x_g_t_by_y, x_g_t_by_w) * 2
        G_loss += self.mse_loss(x_g_t_by_z, x_g_t_by_w)  # + self.ssim_loss(x_g_t_by_z, x_g_t_by_w) * 2

        G_loss += self.mse_loss(y_g, y_g_t_by_x) * 2  # + self.ssim_loss(y_g, y_g_t_by_x) * 2
        G_loss += self.mse_loss(y_g, y_g_t_by_z) * 2  # + self.ssim_loss(y_g, y_g_t_by_z) * 2
        G_loss += self.mse_loss(y_g, y_g_t_by_w) * 2  # + self.ssim_loss(y_g, y_g_t_by_w) * 2
        G_loss += self.mse_loss(y_g_t_by_x, y_g_t_by_z)  # + self.ssim_loss(y_g_t_by_x, y_g_t_by_z) * 2
        G_loss += self.mse_loss(y_g_t_by_x, y_g_t_by_w)  # + self.ssim_loss(y_g_t_by_x, y_g_t_by_w) * 2
        G_loss += self.mse_loss(y_g_t_by_z, y_g_t_by_w)  # + self.ssim_loss(y_g_t_by_z, y_g_t_by_w) * 2

        G_loss += self.mse_loss(z_g, z_g_t_by_x) * 2  # + self.ssim_loss(z_g, z_g_t_by_x) * 2
        G_loss += self.mse_loss(z_g, z_g_t_by_y) * 2  # + self.ssim_loss(z_g, z_g_t_by_y) * 2
        G_loss += self.mse_loss(z_g, z_g_t_by_w) * 2  # + self.ssim_loss(z_g, z_g_t_by_w) * 2
        G_loss += self.mse_loss(z_g_t_by_x, z_g_t_by_y)  # + self.ssim_loss(z_g_t_by_x, z_g_t_by_y) * 2
        G_loss += self.mse_loss(z_g_t_by_x, z_g_t_by_w)  # + self.ssim_loss(z_g_t_by_x, z_g_t_by_w) * 2
        G_loss += self.mse_loss(z_g_t_by_y, z_g_t_by_w)  # + self.ssim_loss(z_g_t_by_y, z_g_t_by_w) * 2

        G_loss += self.mse_loss(w_g, w_g_t_by_x) * 2  # + self.ssim_loss(w_g, w_g_t_by_x) * 2
        G_loss += self.mse_loss(w_g, w_g_t_by_y) * 2  # + self.ssim_loss(w_g, w_g_t_by_y) * 2
        G_loss += self.mse_loss(w_g, w_g_t_by_z) * 2  # + self.ssim_loss(w_g, w_g_t_by_z) * 2
        G_loss += self.mse_loss(w_g_t_by_x, w_g_t_by_y)  # + self.ssim_loss(w_g_t_by_x, w_g_t_by_y) * 2
        G_loss += self.mse_loss(w_g_t_by_x, w_g_t_by_z)  # + self.ssim_loss(w_g_t_by_x, w_g_t_by_z) * 2
        G_loss += self.mse_loss(w_g_t_by_y, w_g_t_by_z)  # + self.ssim_loss(w_g_t_by_y, w_g_t_by_z) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * mask) * 1.5
        G_loss += self.mse_loss(0.0, y_g * mask) * 1.5

        G_loss += self.mse_loss(0.0, x_g_t_by_y * mask) * 1.5
        G_loss += self.mse_loss(0.0, x_g_t_by_z * mask) * 1.5
        G_loss += self.mse_loss(0.0, x_g_t_by_w * mask) * 1.5

        G_loss += self.mse_loss(0.0, y_g_t_by_x * mask) * 1.5
        G_loss += self.mse_loss(0.0, y_g_t_by_z * mask) * 1.5
        G_loss += self.mse_loss(0.0, y_g_t_by_w * mask) * 1.5

        G_loss += self.mse_loss(0.0, z_g_t_by_x * mask) * 1.5
        G_loss += self.mse_loss(0.0, z_g_t_by_y * mask) * 1.5
        G_loss += self.mse_loss(0.0, z_g_t_by_w * mask) * 1.5

        G_loss += self.mse_loss(0.0, w_g_t_by_x * mask) * 1.5
        G_loss += self.mse_loss(0.0, w_g_t_by_y * mask) * 1.5
        G_loss += self.mse_loss(0.0, w_g_t_by_z * mask) * 1.5

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5
        G_loss += self.mse_loss(y, y_r) * 5
        G_loss += self.mse_loss(z, z_r) * 5
        G_loss += self.mse_loss(w, w_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_r_c_by_y) * 10
        G_loss += self.mse_loss(x, x_r_c_by_z) * 10
        G_loss += self.mse_loss(x, x_r_c_by_w) * 10
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_z) * 2
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_w) * 2
        G_loss += self.mse_loss(x_r_c_by_z, x_r_c_by_w) * 2

        G_loss += self.mse_loss(y, y_r_c_by_x) * 10
        G_loss += self.mse_loss(y, y_r_c_by_z) * 10
        G_loss += self.mse_loss(y, y_r_c_by_w) * 10
        G_loss += self.mse_loss(y_r_c_by_x, y_r_c_by_z) * 2
        G_loss += self.mse_loss(y_r_c_by_x, y_r_c_by_w) * 2
        G_loss += self.mse_loss(y_r_c_by_z, y_r_c_by_w) * 2

        G_loss += self.mse_loss(z, z_r_c_by_x) * 10
        G_loss += self.mse_loss(z, z_r_c_by_y) * 10
        G_loss += self.mse_loss(z, z_r_c_by_w) * 10
        G_loss += self.mse_loss(z_r_c_by_x, z_r_c_by_y) * 2
        G_loss += self.mse_loss(z_r_c_by_x, z_r_c_by_w) * 2
        G_loss += self.mse_loss(z_r_c_by_y, z_r_c_by_w) * 2

        G_loss += self.mse_loss(w, w_r_c_by_x) * 10
        G_loss += self.mse_loss(w, w_r_c_by_y) * 10
        G_loss += self.mse_loss(w, w_r_c_by_z) * 10
        G_loss += self.mse_loss(w_r_c_by_x, w_r_c_by_y) * 2
        G_loss += self.mse_loss(w_r_c_by_x, w_r_c_by_z) * 2
        G_loss += self.mse_loss(w_r_c_by_y, w_r_c_by_z) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_t_by_y * mask_y)
        G_loss += self.mse_loss(0.0, x_t_by_z * mask_z)
        G_loss += self.mse_loss(0.0, x_t_by_w * mask_w)

        G_loss += self.mse_loss(0.0, y_t_by_x * mask_x)
        G_loss += self.mse_loss(0.0, y_t_by_z * mask_z)
        G_loss += self.mse_loss(0.0, y_t_by_w * mask_w)

        G_loss += self.mse_loss(0.0, z_t_by_x * mask_x)
        G_loss += self.mse_loss(0.0, z_t_by_y * mask_y)
        G_loss += self.mse_loss(0.0, z_t_by_w * mask_w)

        G_loss += self.mse_loss(0.0, w_t_by_x * mask_x)
        G_loss += self.mse_loss(0.0, w_t_by_y * mask_y)
        G_loss += self.mse_loss(0.0, w_t_by_z * mask_z)

        G_loss += self.mse_loss(0.0, x_r * mask_x) * 0.5
        G_loss += self.mse_loss(0.0, y_r * mask_y) * 0.5
        G_loss += self.mse_loss(0.0, z_r * mask_z) * 0.5
        G_loss += self.mse_loss(0.0, w_r * mask_w) * 0.5

        # 通过解码器生成X模态与Y模态图的编码与X模态与Y模态图经过编码器得到的编码的自监督语义一致性损失
        G_loss += self.mse_loss(code_rm, code_x_g)
        G_loss += self.mse_loss(code_rm, code_y_g)
        G_loss += self.mse_loss(code_rm, code_z_g)
        G_loss += self.mse_loss(code_rm, code_w_g)

        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.2
        G_loss += self.mse_loss(code_x_g, code_z_g) * 0.2
        G_loss += self.mse_loss(code_x_g, code_w_g) * 0.2
        G_loss += self.mse_loss(code_y_g, code_z_g) * 0.2
        G_loss += self.mse_loss(code_y_g, code_w_g) * 0.2
        G_loss += self.mse_loss(code_z_g, code_w_g) * 0.2

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_z_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_w_t_by_x) * 5
        G_loss += self.mse_loss(code_y_t_by_x, code_z_t_by_x)
        G_loss += self.mse_loss(code_y_t_by_x, code_w_t_by_x)
        G_loss += self.mse_loss(code_z_t_by_x, code_w_t_by_x)

        G_loss += self.mse_loss(code_y, code_x_t_by_y) * 5
        G_loss += self.mse_loss(code_y, code_z_t_by_y) * 5
        G_loss += self.mse_loss(code_y, code_w_t_by_y) * 5
        G_loss += self.mse_loss(code_x_t_by_y, code_z_t_by_y)
        G_loss += self.mse_loss(code_x_t_by_y, code_w_t_by_y)
        G_loss += self.mse_loss(code_z_t_by_y, code_w_t_by_y)

        G_loss += self.mse_loss(code_z, code_x_t_by_z) * 5
        G_loss += self.mse_loss(code_z, code_y_t_by_z) * 5
        G_loss += self.mse_loss(code_z, code_w_t_by_z) * 5
        G_loss += self.mse_loss(code_x_t_by_z, code_y_t_by_z)
        G_loss += self.mse_loss(code_x_t_by_z, code_w_t_by_z)
        G_loss += self.mse_loss(code_y_t_by_z, code_w_t_by_z)

        G_loss += self.mse_loss(code_w, code_x_t_by_w) * 5
        G_loss += self.mse_loss(code_w, code_y_t_by_w) * 5
        G_loss += self.mse_loss(code_w, code_z_t_by_w) * 5
        G_loss += self.mse_loss(code_x_t_by_w, code_y_t_by_w)
        G_loss += self.mse_loss(code_x_t_by_w, code_z_t_by_w)
        G_loss += self.mse_loss(code_y_t_by_w, code_z_t_by_w)


        self.image_list["mask"] = mask
        self.image_list["f"] = f
        self.prob_list["label_expand"] = label_expand
        self.prob_list["f_rm_expand"] = f_rm_expand
        self.image_list["l"] = l
        self.image_list["l_x"] = l_x
        self.image_list["l_y"] = l_y
        self.image_list["l_z"] = l_z
        self.image_list["l_w"] = l_w
        self.image_list["x"] = x
        self.image_list["y"] = y
        self.image_list["z"] = z
        self.image_list["w"] = w

        self.code_list["code_rm"] = code_rm

        self.prob_list["l_g_prob"] = l_g_prob
        self.image_list["l_g"] = l_g
        self.image_list["x_g"] = x_g
        self.image_list["y_g"] = y_g
        self.image_list["z_g"] = z_g
        self.image_list["w_g"] = w_g

        self.code_list["code_x_g"] = code_x_g
        self.code_list["code_y_g"] = code_y_g
        self.code_list["code_z_g"] = code_z_g
        self.code_list["code_w_g"] = code_w_g

        self.prob_list["l_g_prob_by_x"] = l_g_prob_by_x
        self.prob_list["l_g_prob_by_y"] = l_g_prob_by_y
        self.prob_list["l_g_prob_by_z"] = l_g_prob_by_z
        self.prob_list["l_g_prob_by_w"] = l_g_prob_by_w
        self.image_list["l_g_by_x"] = l_g_by_x
        self.image_list["l_g_by_y"] = l_g_by_y
        self.image_list["l_g_by_z"] = l_g_by_z
        self.image_list["l_g_by_w"] = l_g_by_w

        self.image_list["f_x_g_r"] = f_x_g_r
        self.image_list["f_y_g_r"] = f_y_g_r
        self.image_list["f_z_g_r"] = f_z_g_r
        self.image_list["f_w_g_r"] = f_w_g_r

        self.image_list["y_g_t_by_x"] = y_g_t_by_x
        self.image_list["z_g_t_by_x"] = z_g_t_by_x
        self.image_list["w_g_t_by_x"] = w_g_t_by_x

        self.image_list["x_g_t_by_y"] = x_g_t_by_y
        self.image_list["z_g_t_by_y"] = z_g_t_by_y
        self.image_list["w_g_t_by_y"] = w_g_t_by_y

        self.image_list["x_g_t_by_z"] = x_g_t_by_z
        self.image_list["y_g_t_by_z"] = y_g_t_by_z
        self.image_list["w_g_t_by_z"] = w_g_t_by_z

        self.image_list["x_g_t_by_w"] = x_g_t_by_w
        self.image_list["y_g_t_by_w"] = y_g_t_by_w
        self.image_list["z_g_t_by_w"] = z_g_t_by_w

        self.prob_list["label_expand_x"] = label_expand_x
        self.prob_list["label_expand_y"] = label_expand_y
        self.prob_list["label_expand_z"] = label_expand_z
        self.prob_list["label_expand_w"] = label_expand_w

        self.image_list["mask_x"] = mask_x
        self.image_list["mask_y"] = mask_y
        self.image_list["mask_z"] = mask_z
        self.image_list["mask_w"] = mask_w

        self.code_list["code_x"] = code_x
        self.code_list["code_y"] = code_y
        self.code_list["code_z"] = code_z
        self.code_list["code_w"] = code_w

        self.prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.prob_list["l_f_prob_by_y"] = l_f_prob_by_y
        self.prob_list["l_f_prob_by_z"] = l_f_prob_by_z
        self.prob_list["l_f_prob_by_w"] = l_f_prob_by_w
        self.image_list["l_f_by_x"] = l_f_by_x
        self.image_list["l_f_by_y"] = l_f_by_y
        self.image_list["l_f_by_z"] = l_f_by_z
        self.image_list["l_f_by_w"] = l_f_by_w

        self.image_list["x_r"] = x_r
        self.image_list["y_r"] = y_r
        self.image_list["z_r"] = z_r
        self.image_list["w_r"] = w_r

        self.image_list["y_t_by_x"] = y_t_by_x
        self.code_list["code_y_t_by_x"] = code_y_t_by_x
        self.image_list["x_r_c_by_y"] = x_r_c_by_y
        self.image_list["z_t_by_x"] = z_t_by_x
        self.code_list["code_z_t_by_x"] = code_z_t_by_x
        self.image_list["x_r_c_by_z"] = x_r_c_by_z
        self.image_list["w_t_by_x"] = w_t_by_x
        self.code_list["code_w_t_by_x"] = code_w_t_by_x
        self.image_list["x_r_c_by_w"] = x_r_c_by_w

        self.image_list["x_t_by_y"] = x_t_by_y
        self.code_list["code_x_t_by_y"] = code_x_t_by_y
        self.image_list["y_r_c_by_x"] = y_r_c_by_x
        self.image_list["z_t_by_y"] = z_t_by_y
        self.code_list["code_z_t_by_y"] = code_z_t_by_y
        self.image_list["y_r_c_by_z"] = y_r_c_by_z
        self.image_list["w_t_by_y"] = w_t_by_y
        self.code_list["code_w_t_by_y"] = code_w_t_by_y
        self.image_list["y_r_c_by_w"] = y_r_c_by_w

        self.image_list["x_t_by_z"] = x_t_by_z
        self.code_list["code_x_t_by_z"] = code_x_t_by_z
        self.image_list["z_r_c_by_x"] = z_r_c_by_x
        self.image_list["y_t_by_z"] = y_t_by_z
        self.code_list["code_y_t_by_z"] = code_y_t_by_z
        self.image_list["z_r_c_by_y"] = z_r_c_by_y
        self.image_list["w_t_by_z"] = w_t_by_z
        self.code_list["code_w_t_by_z"] = code_w_t_by_z
        self.image_list["z_r_c_by_w"] = z_r_c_by_w

        self.image_list["x_t_by_w"] = x_t_by_w
        self.code_list["code_x_t_by_w"] = code_x_t_by_w
        self.image_list["w_r_c_by_x"] = w_r_c_by_x
        self.image_list["y_t_by_w"] = y_t_by_w
        self.code_list["code_y_t_by_w"] = code_y_t_by_w
        self.image_list["w_r_c_by_y"] = w_r_c_by_y
        self.image_list["z_t_by_w"] = z_t_by_w
        self.code_list["code_z_t_by_w"] = code_z_t_by_w
        self.image_list["w_r_c_by_z"] = w_r_c_by_z

        self.judge_list["j_x_g"], self.judge_list["j_x_g_c"] = j_x_g, j_x_g_c
        self.judge_list["j_y_g"], self.judge_list["j_y_g_c"] = j_y_g, j_y_g_c
        self.judge_list["j_z_g"], self.judge_list["j_z_g_c"] = j_z_g, j_z_g_c
        self.judge_list["j_w_g"], self.judge_list["j_w_g_c"] = j_w_g, j_w_g_c

        self.judge_list["j_x"], self.judge_list["j_x_c"] = j_x, j_x_c
        self.judge_list["j_y"], self.judge_list["j_y_c"] = j_y, j_y_c
        self.judge_list["j_z"], self.judge_list["j_z_c"] = j_z, j_z_c
        self.judge_list["j_w"], self.judge_list["j_w_c"] = j_w, j_w_c

        self.judge_list["j_x_t_by_y"], self.judge_list["j_x_t_c_by_y"] = j_x_t_by_y, j_x_t_c_by_y
        self.judge_list["j_x_t_by_z"], self.judge_list["j_x_t_c_by_z"] = j_x_t_by_z, j_x_t_c_by_z
        self.judge_list["j_x_t_by_w"], self.judge_list["j_x_t_c_by_w"] = j_x_t_by_w, j_x_t_c_by_w

        self.judge_list["j_y_t_by_x"], self.judge_list["j_y_t_c_by_x"] = j_y_t_by_x, j_y_t_c_by_x
        self.judge_list["j_y_t_by_z"], self.judge_list["j_y_t_c_by_z"] = j_y_t_by_z, j_y_t_c_by_z
        self.judge_list["j_y_t_by_w"], self.judge_list["j_y_t_c_by_w"] = j_y_t_by_w, j_y_t_c_by_w

        self.judge_list["j_z_t_by_x"], self.judge_list["j_z_t_c_by_x"] = j_z_t_by_x, j_z_t_c_by_x
        self.judge_list["j_z_t_by_y"], self.judge_list["j_z_t_c_by_y"] = j_z_t_by_y, j_z_t_c_by_y
        self.judge_list["j_z_t_by_w"], self.judge_list["j_z_t_c_by_w"] = j_z_t_by_w, j_z_t_c_by_w

        self.judge_list["j_w_t_by_x"], self.judge_list["j_w_t_c_by_x"] = j_w_t_by_x, j_w_t_c_by_x
        self.judge_list["j_w_t_by_y"], self.judge_list["j_w_t_c_by_y"] = j_w_t_by_y, j_w_t_c_by_y
        self.judge_list["j_w_t_by_z"], self.judge_list["j_w_t_c_by_z"] = j_w_t_by_z, j_w_t_c_by_z

        self.judge_list["j_code_x"] = j_code_x
        self.judge_list["j_code_y"] = j_code_y
        self.judge_list["j_code_z"] = j_code_z
        self.judge_list["j_code_w"] = j_code_w
        self.judge_list["j_code_rm"] = j_code_rm

        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.DC_L.variables
                + self.EC_M.variables
                + self.DC_M.variables
            ,
                self.D_M.variables +
                self.FD_R.variables
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
            tf.summary.image('judge/' + key, judge_dirct[key])

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_dirct):
        self.name_code_list_true = ["code_rm", "code_rm", "code_rm", "code_rm",
                                    "code_x", "code_x", "code_x",
                                    "code_y", "code_y", "code_y",
                                    "code_z", "code_z", "code_z",
                                    "code_w", "code_w", "code_w"]
        self.name_code_list_false = ["code_x_g", "code_y_g", "code_z_g", "code_w_g",
                                     "code_y_t_by_x", "code_z_t_by_x", "code_w_t_by_x",
                                     "code_x_t_by_y", "code_z_t_by_y", "code_w_t_by_y",
                                     "code_x_t_by_z", "code_y_t_by_z", "code_w_t_by_z",
                                     "code_x_t_by_w", "code_y_t_by_w", "code_z_t_by_w"]
        ssim_list = []
        for i in range(len(self.name_code_list_true)):
            ssim_list.append(
                self.SSIM(code_dirct[self.name_code_list_true[i]], code_dirct[self.name_code_list_false[i]]))
        return ssim_list

    def evaluation_code_summary(self, ssim_list):
        for i in range(len(self.name_code_list_true)):
            tf.summary.scalar(
                "evaluation_code/" + self.name_code_list_true[i] + "__VS__" + self.name_code_list_false[i],
                ssim_list[i])

    def evaluation(self, image_dirct):
        self.name_list_true = ["l", "l", "l", "l", "l",
                               "l_x", "l_y", "l_z", "l_w",
                               "x", "y", "z", "w",
                               "x_g", "x_g", "x_g",
                               "y_g", "y_g", "y_g",
                               "z_g", "z_g", "z_g",
                               "w_g", "w_g", "w_g", ]
        self.name_list_false = ["l_g", "l_g_by_x", "l_g_by_y", "l_g_by_z", "l_g_by_w",
                                "l_f_by_x", "l_f_by_y", "l_f_by_z", "l_f_by_w",
                                "x_r", "y_r", "z_r", "w_r",
                                "x_g_t_by_y", "x_g_t_by_z", "x_g_t_by_w",
                                "y_g_t_by_x", "y_g_t_by_z", "y_g_t_by_w",
                                "z_g_t_by_x", "z_g_t_by_y", "z_g_t_by_w",
                                "w_g_t_by_x", "w_g_t_by_y", "w_g_t_by_z",
                                ]
        ssim_list = []
        for i in range(len(self.name_list_true)):
            ssim_list.append(self.SSIM(image_dirct[self.name_list_true[i]], image_dirct[self.name_list_false[i]]))
        return ssim_list

    def evaluation_summary(self, ssim_list):
        for i in range(len(self.name_list_true)):
            tf.summary.scalar("evaluation/" + self.name_list_true[i] + "__VS__" + self.name_list_false[i], ssim_list[i])

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
