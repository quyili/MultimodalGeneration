# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from unet import Unet


class GAN:
    def __init__(self,
                 image_size,
                 learning_rate=2e-5,
                 batch_size=1,
                 ngf=64,
                 ):
        """
           Args:
             input_size：list [N, H, W, C]
             batch_size: integer, batch size
             learning_rate: float, initial learning rate for Adam
             ngf: number of base gen filters in conv layer
        """
        self.learning_rate = learning_rate
        self.input_shape = [int(batch_size / 4), image_size[0], image_size[1], image_size[2]]
        self.code_shape = [int(batch_size / 4), int(image_size[0] / 4), int(image_size[1] / 4), 4]
        self.ones = tf.ones(self.input_shape, name="ones")
        self.ones_code = tf.ones(self.code_shape, name="ones_code")
        self.image_list = {}
        self.prob_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.G_L_X = Unet('G_L_X', ngf=ngf, output_channl=5)
        self.G_L_Y = Unet('G_L_Y', ngf=ngf, output_channl=5)
        self.G_L_Z = Unet('G_L_Z', ngf=ngf, output_channl=5)
        self.G_L_W = Unet('G_L_W', ngf=ngf, output_channl=5)

        self.G_T = Unet('G_T', ngf=ngf)
        self.D_T = Discriminator('D_T', ngf=ngf, output=2)

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
        f = f * l_mask
        return f

    def segmentation(self, x, EC_L, DC_L):
        l_prob = DC_L(EC_L(x))
        l_f = tf.reshape(tf.cast(tf.argmax(l_prob, axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        return l_prob, l_f

    def model(self, l_x, l_y, l_z, l_w, x, y, z, w):
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)

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

        l_g_prob_by_x = self.G_L_X(x)
        l_g_prob_by_y = self.G_L_Y(y)
        l_g_prob_by_z = self.G_L_Z(z)
        l_g_prob_by_w = self.G_L_W(w)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_z = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_z, axis=-1), dtype=tf.float32), shape=self.input_shape)
        l_g_by_w = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_w, axis=-1), dtype=tf.float32), shape=self.input_shape)

        y_t_by_x = self.G_T(x, cy_code)
        z_t_by_x = self.G_T(x, cz_code)
        w_t_by_x = self.G_T(x, cw_code)

        x_t_by_y = self.G_T(y, cx_code)
        z_t_by_y = self.G_T(y, cz_code)
        w_t_by_y = self.G_T(y, cw_code)

        x_t_by_z = self.G_T(z, cx_code)
        y_t_by_z = self.G_T(z, cy_code)
        w_t_by_z = self.G_T(z, cw_code)

        x_t_by_w = self.G_T(w, cx_code)
        y_t_by_w = self.G_T(w, cy_code)
        z_t_by_w = self.G_T(w, cz_code)

        x_r_c_by_y = self.G_T(y_t_by_x, cx_code)
        x_r_c_by_z = self.G_T(z_t_by_x, cx_code)
        x_r_c_by_w = self.G_T(w_t_by_x, cx_code)

        y_r_c_by_x = self.G_T(x_t_by_y, cy_code)
        y_r_c_by_z = self.G_T(z_t_by_y, cy_code)
        y_r_c_by_w = self.G_T(w_t_by_y, cy_code)

        z_r_c_by_x = self.G_T(x_t_by_z, cz_code)
        z_r_c_by_y = self.G_T(y_t_by_z, cz_code)
        z_r_c_by_w = self.G_T(w_t_by_z, cz_code)

        w_r_c_by_x = self.G_T(x_t_by_w, cw_code)
        w_r_c_by_y = self.G_T(y_t_by_w, cw_code)
        w_r_c_by_z = self.G_T(z_t_by_w, cw_code)

        j_x, j_x_c = self.D_T(x)
        j_y, j_y_c = self.D_T(y)
        j_z, j_z_c = self.D_T(z)
        j_w, j_w_c = self.D_T(w)

        j_x_t_by_y, j_x_t_c_by_y = self.D_T(x_t_by_y)
        j_x_t_by_z, j_x_t_c_by_z = self.D_T(x_t_by_z)
        j_x_t_by_w, j_x_t_c_by_w = self.D_T(x_t_by_w)

        j_y_t_by_x, j_y_t_c_by_x = self.D_T(y_t_by_x)
        j_y_t_by_z, j_y_t_c_by_z = self.D_T(y_t_by_z)
        j_y_t_by_w, j_y_t_c_by_w = self.D_T(y_t_by_w)

        j_z_t_by_x, j_z_t_c_by_x = self.D_T(z_t_by_x)
        j_z_t_by_y, j_z_t_c_by_y = self.D_T(z_t_by_y)
        j_z_t_by_w, j_z_t_c_by_w = self.D_T(z_t_by_w)

        j_w_t_by_x, j_w_t_c_by_x = self.D_T(w_t_by_x)
        j_w_t_by_y, j_w_t_c_by_y = self.D_T(w_t_by_y)
        j_w_t_by_z, j_w_t_c_by_z = self.D_T(w_t_by_z)

        D_loss = 0.0
        G_loss = 0.0
        S_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_x_t_by_y, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_x_t_by_z, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_x_t_by_w, 0.0) * 5 * 2
        G_loss += self.mse_loss(j_x_t_by_y, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_x_t_by_z, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_x_t_by_w, 1.0) * 5 * 2

        D_loss += self.mse_loss(j_y, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_y_t_by_x, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_y_t_by_z, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_y_t_by_w, 0.0) * 5 * 2
        G_loss += self.mse_loss(j_y_t_by_x, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_y_t_by_z, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_y_t_by_w, 1.0) * 5 * 2

        D_loss += self.mse_loss(j_z, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_z_t_by_x, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_z_t_by_y, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_z_t_by_w, 0.0) * 5 * 2
        G_loss += self.mse_loss(j_z_t_by_x, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_z_t_by_y, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_z_t_by_w, 1.0) * 5 * 2

        D_loss += self.mse_loss(j_w, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_w_t_by_x, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_w_t_by_y, 0.0) * 5 * 2
        D_loss += self.mse_loss(j_w_t_by_z, 0.0) * 5 * 2
        G_loss += self.mse_loss(j_w_t_by_x, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_w_t_by_y, 1.0) * 5 * 2
        G_loss += self.mse_loss(j_w_t_by_z, 1.0) * 5 * 2

        D_loss += self.mse_loss(j_x_c, cx) * 50 * 2
        D_loss += self.mse_loss(j_y_c, cy) * 50 * 2
        D_loss += self.mse_loss(j_z_c, cz) * 50 * 2
        D_loss += self.mse_loss(j_w_c, cw) * 50 * 2

        G_loss += self.mse_loss(j_x_t_c_by_y, cx) * 50 * 2
        G_loss += self.mse_loss(j_x_t_c_by_z, cx) * 50 * 2
        G_loss += self.mse_loss(j_x_t_c_by_w, cx) * 50 * 2

        G_loss += self.mse_loss(j_y_t_c_by_x, cy) * 50 * 2
        G_loss += self.mse_loss(j_y_t_c_by_z, cy) * 50 * 2
        G_loss += self.mse_loss(j_y_t_c_by_w, cy) * 50 * 2

        G_loss += self.mse_loss(j_z_t_c_by_x, cz) * 50 * 2
        G_loss += self.mse_loss(j_z_t_c_by_y, cz) * 50 * 2
        G_loss += self.mse_loss(j_z_t_c_by_w, cz) * 50 * 2

        G_loss += self.mse_loss(j_w_t_c_by_x, cw) * 50 * 2
        G_loss += self.mse_loss(j_w_t_c_by_y, cw) * 50 * 2
        G_loss += self.mse_loss(j_w_t_c_by_z, cw) * 50 * 2

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

        S_loss += self.mse_loss(label_expand_x[:, :, :, 0],
                                l_g_prob_by_x[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 1],
                                  l_g_prob_by_x[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 2],
                                  l_g_prob_by_x[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 3],
                                  l_g_prob_by_x[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 4],
                                  l_g_prob_by_x[:, :, :, 4]) * 25 * 5

        S_loss += self.mse_loss(label_expand_y[:, :, :, 0],
                                l_g_prob_by_y[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 1],
                                  l_g_prob_by_y[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 2],
                                  l_g_prob_by_y[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 3],
                                  l_g_prob_by_y[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 4],
                                  l_g_prob_by_y[:, :, :, 4]) * 25 * 5

        S_loss += self.mse_loss(label_expand_z[:, :, :, 0],
                                l_g_prob_by_z[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(label_expand_z[:, :, :, 1],
                                  l_g_prob_by_z[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(label_expand_z[:, :, :, 2],
                                  l_g_prob_by_z[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(label_expand_z[:, :, :, 3],
                                  l_g_prob_by_z[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(label_expand_z[:, :, :, 4],
                                  l_g_prob_by_z[:, :, :, 4]) * 25 * 5

        S_loss += self.mse_loss(label_expand_w[:, :, :, 0],
                                l_g_prob_by_w[:, :, :, 0]) * 0.5 * 5 \
                  + self.mse_loss(label_expand_w[:, :, :, 1],
                                  l_g_prob_by_w[:, :, :, 1]) * 5 * 5 \
                  + self.mse_loss(label_expand_w[:, :, :, 2],
                                  l_g_prob_by_w[:, :, :, 2]) * 25 * 5 \
                  + self.mse_loss(label_expand_w[:, :, :, 3],
                                  l_g_prob_by_w[:, :, :, 3]) * 25 * 5 \
                  + self.mse_loss(label_expand_w[:, :, :, 4],
                                  l_g_prob_by_w[:, :, :, 4]) * 25 * 5

        self.image_list["l_x"] = l_x * 0.25
        self.image_list["l_y"] = l_y * 0.25
        self.image_list["l_z"] = l_z * 0.25
        self.image_list["l_w"] = l_w * 0.25
        self.image_list["x"] = x
        self.image_list["y"] = y
        self.image_list["z"] = z
        self.image_list["w"] = w

        self.prob_list["l_g_prob_by_x"] = l_g_prob_by_x
        self.prob_list["l_g_prob_by_y"] = l_g_prob_by_y
        self.prob_list["l_g_prob_by_z"] = l_g_prob_by_z
        self.prob_list["l_g_prob_by_w"] = l_g_prob_by_w
        self.image_list["l_g_by_x"] = l_g_by_x * 0.25
        self.image_list["l_g_by_y"] = l_g_by_y * 0.25
        self.image_list["l_g_by_z"] = l_g_by_z * 0.25
        self.image_list["l_g_by_w"] = l_g_by_w * 0.25

        self.image_list["y_t_by_x"] = y_t_by_x
        self.image_list["z_t_by_x"] = z_t_by_x
        self.image_list["w_t_by_x"] = w_t_by_x

        self.image_list["x_t_by_y"] = x_t_by_y
        self.image_list["z_t_by_y"] = z_t_by_y
        self.image_list["w_t_by_y"] = w_t_by_y

        self.image_list["x_t_by_z"] = x_t_by_z
        self.image_list["y_t_by_z"] = y_t_by_z
        self.image_list["w_t_by_z"] = w_t_by_z

        self.image_list["x_t_by_w"] = x_t_by_w
        self.image_list["y_t_by_w"] = y_t_by_w
        self.image_list["z_t_by_w"] = z_t_by_w

        self.prob_list["label_expand_x"] = label_expand_x
        self.prob_list["label_expand_y"] = label_expand_y
        self.prob_list["label_expand_z"] = label_expand_z
        self.prob_list["label_expand_w"] = label_expand_w

        self.image_list["y_t_by_x"] = y_t_by_x
        self.image_list["x_r_c_by_y"] = x_r_c_by_y
        self.image_list["z_t_by_x"] = z_t_by_x
        self.image_list["x_r_c_by_z"] = x_r_c_by_z
        self.image_list["w_t_by_x"] = w_t_by_x
        self.image_list["x_r_c_by_w"] = x_r_c_by_w

        self.image_list["x_t_by_y"] = x_t_by_y
        self.image_list["y_r_c_by_x"] = y_r_c_by_x
        self.image_list["z_t_by_y"] = z_t_by_y
        self.image_list["y_r_c_by_z"] = y_r_c_by_z
        self.image_list["w_t_by_y"] = w_t_by_y
        self.image_list["y_r_c_by_w"] = y_r_c_by_w

        self.image_list["x_t_by_z"] = x_t_by_z
        self.image_list["z_r_c_by_x"] = z_r_c_by_x
        self.image_list["y_t_by_z"] = y_t_by_z
        self.image_list["z_r_c_by_y"] = z_r_c_by_y
        self.image_list["w_t_by_z"] = w_t_by_z
        self.image_list["z_r_c_by_w"] = z_r_c_by_w

        self.image_list["x_t_by_w"] = x_t_by_w
        self.image_list["w_r_c_by_x"] = w_r_c_by_x
        self.image_list["y_t_by_w"] = y_t_by_w
        self.image_list["w_r_c_by_y"] = w_r_c_by_y
        self.image_list["z_t_by_w"] = z_t_by_w
        self.image_list["w_r_c_by_z"] = w_r_c_by_z

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

        loss_list = [G_loss, D_loss, S_loss]

        return loss_list

    def get_variables(self):
        return [self.G_T.variables
            ,
                self.D_T.variables
            ,
                self.G_L_X.variables
                + self.G_L_Y.variables
                + self.G_L_Z.variables
                + self.G_L_W.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')
        D_optimizer = make_optimizer(name='Adam_D')
        S_optimizer = make_optimizer(name='Adam_S')

        return G_optimizer, D_optimizer, S_optimizer

    def mse_loss(self, x, y):
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    def ssim_loss(self, x, y):
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
