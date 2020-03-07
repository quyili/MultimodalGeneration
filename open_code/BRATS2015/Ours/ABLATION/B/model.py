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

        self.G_X = Unet('G_X', ngf=ngf)
        self.D_X = Discriminator('D_X', ngf=ngf, output=2)

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

    def model(self, l, l_m, m, x, y, z, w):
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)

        mask = self.get_mask(m)
        f = self.get_f(m)  # M->F
        f = self.remove_l(l_m, f)
        self.tenaor_name["l"] = str(l)
        self.tenaor_name["f"] = str(f)
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                  shape=[self.input_shape[0], self.input_shape[1],
                                         self.input_shape[2], 5])
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

        x_g = self.G_X(f_rm_expand, cx_code)
        y_g = self.G_X(f_rm_expand, cy_code)
        z_g = self.G_X(f_rm_expand, cz_code)
        w_g = self.G_X(f_rm_expand, cw_code)

        self.tenaor_name["x_g"] = str(x_g)
        self.tenaor_name["y_g"] = str(y_g)
        self.tenaor_name["z_g"] = str(z_g)
        self.tenaor_name["w_g"] = str(w_g)

        j_x_g, j_x_g_c = self.D_X(x_g)
        j_y_g, j_y_g_c = self.D_X(y_g)
        j_z_g, j_z_g_c = self.D_X(z_g)
        j_w_g, j_w_g_c = self.D_X(w_g)

        j_x, j_x_c = self.D_X(x)
        j_y, j_y_c = self.D_X(y)
        j_z, j_z_c = self.D_X(z)
        j_w, j_w_c = self.D_X(w)

        D_loss = 0.0
        G_loss = 0.0
        D_loss += self.mse_loss(j_x, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_x_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_x_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_y, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_y_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_y_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_z, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_z_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_z_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_w, 1.0) * 50 * 2
        D_loss += self.mse_loss(j_w_g, 0.0) * 35 * 2
        G_loss += self.mse_loss(j_w_g, 1.0) * 35 * 2

        D_loss += self.mse_loss(j_x_c, cx) * 50 * 2
        D_loss += self.mse_loss(j_y_c, cy) * 50 * 2
        D_loss += self.mse_loss(j_z_c, cz) * 50 * 2
        D_loss += self.mse_loss(j_w_c, cw) * 50 * 2

        G_loss += self.mse_loss(j_x_g_c, cx) * 50 * 2
        G_loss += self.mse_loss(j_y_g_c, cy) * 50 * 2
        G_loss += self.mse_loss(j_z_g_c, cz) * 50 * 2
        G_loss += self.mse_loss(j_w_g_c, cw) * 50 * 2

        self.image_list["mask"] = mask
        self.image_list["f"] = f
        self.prob_list["label_expand"] = label_expand
        self.prob_list["f_rm_expand"] = f_rm_expand
        self.image_list["l"] = l*0.25

        self.image_list["x_g"] = x_g
        self.image_list["y_g"] = y_g
        self.image_list["z_g"] = z_g
        self.image_list["w_g"] = w_g

        self.judge_list["j_x_g"], self.judge_list["j_x_g_c"] = j_x_g, j_x_g_c
        self.judge_list["j_y_g"], self.judge_list["j_y_g_c"] = j_y_g, j_y_g_c
        self.judge_list["j_z_g"], self.judge_list["j_z_g_c"] = j_z_g, j_z_g_c
        self.judge_list["j_w_g"], self.judge_list["j_w_g_c"] = j_w_g, j_w_g_c

        self.judge_list["j_x"], self.judge_list["j_x_c"] = j_x, j_x_c
        self.judge_list["j_y"], self.judge_list["j_y_c"] = j_y, j_y_c
        self.judge_list["j_z"], self.judge_list["j_z_c"] = j_z, j_z_c
        self.judge_list["j_w"], self.judge_list["j_w_c"] = j_w, j_w_c

        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.G_X.variables
            ,
                self.D_X.variables
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
