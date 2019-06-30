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
        self.image_list = {}
        self.prob_list = {}
        self.code_list = {}
        self.judge_list = {}
        self.tenaor_name = {}

        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)

        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.EC_Z = Encoder('EC_Z', ngf=ngf)
        self.EC_W = Encoder('EC_W', ngf=ngf)

        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_Z = Decoder('DC_Z', ngf=ngf)
        self.DC_W = Decoder('DC_W', ngf=ngf)

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

    def get_mask(self, x):
        mask = 1.0 - self.ones * tf.cast(x > 0.02, dtype="float32")
        return mask

    def model(self, l, m, l_x, l_y, l_z, l_w, x, y, z, w):
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        self.image_list["mask"] = self.get_mask(m)
        self.image_list["f"] = self.get_f(m)  # M->F
        self.prob_list["label_expand"] = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=5),
                                                    shape=[self.input_shape[0], self.input_shape[1],
                                                           self.input_shape[2], 5])
        self.prob_list["f_rm_expand"] = tf.concat([
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * self.prob_list["label_expand"][:, :, :, 0],
                       shape=self.input_shape) + self.image_list["f"] * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * self.prob_list["label_expand"][:, :, :, 1],
                       shape=self.input_shape) + self.image_list["f"] * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * self.prob_list["label_expand"][:, :, :, 2],
                       shape=self.input_shape) + self.image_list["f"] * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * self.prob_list["label_expand"][:, :, :, 3],
                       shape=self.input_shape) + self.image_list["f"] * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * self.prob_list["label_expand"][:, :, :, 4],
                       shape=self.input_shape) + self.image_list["f"] * 0.8],
            axis=-1)

        # TODO input f
        # self.image_list["f"] = f
        l = l * 0.25
        self.image_list["l"] = l
        self.image_list["l_x"] = l_x
        self.image_list["l_y"] = l_y
        self.image_list["l_z"] = l_z
        self.image_list["l_w"] = l_w
        self.image_list["x"] = x
        self.image_list["y"] = y
        self.image_list["z"] = z
        self.image_list["w"] = w

        self.code_list["code_rm"] = self.EC_R(self.prob_list["f_rm_expand"])

        self.prob_list["l_g_prob"] = self.DC_L(self.code_list["code_rm"])
        self.image_list["l_g"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_g_prob"], axis=-1), dtype=tf.float32) * 0.25, shape=self.input_shape)
        self.image_list["x_g"] = self.DC_X(self.code_list["code_rm"])
        self.image_list["y_g"] = self.DC_Y(self.code_list["code_rm"])
        self.image_list["z_g"] = self.DC_Z(self.code_list["code_rm"])
        self.image_list["w_g"] = self.DC_W(self.code_list["code_rm"])

        self.code_list["code_x_g"] = self.EC_X(self.image_list["x_g"])
        self.code_list["code_y_g"] = self.EC_Y(self.image_list["y_g"])
        self.code_list["code_z_g"] = self.EC_Z(self.image_list["z_g"])
        self.code_list["code_w_g"] = self.EC_W(self.image_list["w_g"])

        self.prob_list["l_g_prob_by_x"] = self.DC_L(self.code_list["code_x_g"])
        self.prob_list["l_g_prob_by_y"] = self.DC_L(self.code_list["code_y_g"])
        self.prob_list["l_g_prob_by_z"] = self.DC_L(self.code_list["code_z_g"])
        self.prob_list["l_g_prob_by_w"] = self.DC_L(self.code_list["code_w_g"])
        self.image_list["l_g_by_x"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_g_prob_by_x"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_g_by_y"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_g_prob_by_y"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_g_by_z"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_g_prob_by_z"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_g_by_w"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_g_prob_by_w"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        self.image_list["f_x_g_r"] = self.get_f(self.image_list["x_g"])
        self.image_list["f_y_g_r"] = self.get_f(self.image_list["y_g"])
        self.image_list["f_z_g_r"] = self.get_f(self.image_list["z_g"])
        self.image_list["f_w_g_r"] = self.get_f(self.image_list["w_g"])

        self.image_list["y_g_t_by_x"] = self.DC_Y(self.code_list["code_x_g"])
        # self.image_list["z_g_t_by_x"] = self.DC_Z(self.code_list["code_x_g"])
        self.image_list["w_g_t_by_x"] = self.DC_W(self.code_list["code_x_g"])

        self.image_list["x_g_t_by_y"] = self.DC_X(self.code_list["code_y_g"])
        self.image_list["z_g_t_by_y"] = self.DC_Z(self.code_list["code_y_g"])
        # self.image_list["w_g_t_by_y"] = self.DC_W(self.code_list["code_y_g"])

        # self.image_list["x_g_t_by_z"] = self.DC_X(self.code_list["code_z_g"])
        self.image_list["y_g_t_by_z"] = self.DC_Y(self.code_list["code_z_g"])
        self.image_list["w_g_t_by_z"] = self.DC_W(self.code_list["code_z_g"])

        self.image_list["x_g_t_by_w"] = self.DC_X(self.code_list["code_w_g"])
        # self.image_list["y_g_t_by_w"] = self.DC_Y(self.code_list["code_w_g"])
        self.image_list["z_g_t_by_w"] = self.DC_Z(self.code_list["code_w_g"])

        self.prob_list["label_expand_x"] = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        self.prob_list["label_expand_y"] = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        self.prob_list["label_expand_z"] = tf.reshape(tf.one_hot(tf.cast(l_z, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])
        self.prob_list["label_expand_w"] = tf.reshape(tf.one_hot(tf.cast(l_w, dtype=tf.int32), axis=-1, depth=5),
                                                      shape=[self.input_shape[0], self.input_shape[1],
                                                             self.input_shape[2], 5])

        self.image_list["mask_x"] = self.get_mask(x)
        self.image_list["mask_y"] = self.get_mask(y)
        self.image_list["mask_z"] = self.get_mask(z)
        self.image_list["mask_w"] = self.get_mask(w)

        self.code_list["code_x"] = self.EC_X(x)
        self.code_list["code_y"] = self.EC_Y(y)
        self.code_list["code_z"] = self.EC_Z(z)
        self.code_list["code_w"] = self.EC_W(w)

        self.prob_list["l_f_prob_by_x"] = self.DC_L(self.code_list["code_x"])
        self.prob_list["l_f_prob_by_y"] = self.DC_L(self.code_list["code_y"])
        self.prob_list["l_f_prob_by_z"] = self.DC_L(self.code_list["code_z"])
        self.prob_list["l_f_prob_by_w"] = self.DC_L(self.code_list["code_w"])
        self.image_list["l_f_by_x"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_f_prob_by_x"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_f_by_y"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_f_prob_by_y"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_f_by_z"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_f_prob_by_z"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        self.image_list["l_f_by_w"] = tf.reshape(
            tf.cast(tf.argmax(self.prob_list["l_f_prob_by_w"], axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        self.image_list["x_r"] = self.DC_X(self.code_list["code_x"])
        self.image_list["y_r"] = self.DC_Y(self.code_list["code_y"])
        self.image_list["z_r"] = self.DC_Z(self.code_list["code_z"])
        self.image_list["w_r"] = self.DC_W(self.code_list["code_w"])

        self.image_list["y_t_by_x"] = self.DC_Y(self.code_list["code_x"])
        self.code_list["code_y_t_by_x"] = self.EC_Y(self.image_list["y_t_by_x"])
        self.image_list["x_r_c_by_y"] = self.DC_X(self.code_list["code_y_t_by_x"])
        # self.image_list["z_t_by_x"] = self.DC_Z(self.code_list["code_x"])
        # self.code_list["code_z_t_by_x"] = self.EC_Z(self.image_list["z_t_by_x"])
        # self.image_list["x_r_c_by_z"] = self.DC_X(self.code_list["code_z_t_by_x"])
        self.image_list["w_t_by_x"] = self.DC_W(self.code_list["code_x"])
        self.code_list["code_w_t_by_x"] = self.EC_W(self.image_list["w_t_by_x"])
        self.image_list["x_r_c_by_w"] = self.DC_X(self.code_list["code_w_t_by_x"])

        self.image_list["x_t_by_y"] = self.DC_X(self.code_list["code_y"])
        self.code_list["code_x_t_by_y"] = self.EC_X(self.image_list["x_t_by_y"])
        self.image_list["y_r_c_by_x"] = self.DC_Y(self.code_list["code_x_t_by_y"])
        self.image_list["z_t_by_y"] = self.DC_Z(self.code_list["code_y"])
        self.code_list["code_z_t_by_y"] = self.EC_Z(self.image_list["x_t_by_y"])
        self.image_list["y_r_c_by_z"] = self.DC_Y(self.code_list["code_z_t_by_y"])
        # self.image_list["w_t_by_y"] = self.DC_W(self.code_list["code_y"])
        # self.code_list["code_w_t_by_y"] = self.EC_W(self.image_list["x_t_by_y"])
        # self.image_list["y_r_c_by_w"] = self.DC_Y(self.code_list["code_w_t_by_y"])

        # self.image_list["x_t_by_z"] = self.DC_X(self.code_list["code_z"])
        # self.code_list["code_x_t_by_z"] = self.EC_X(self.image_list["x_t_by_z"])
        # self.image_list["z_r_c_by_x"] = self.DC_Z(self.code_list["code_x_t_by_z"])
        self.image_list["y_t_by_z"] = self.DC_Y(self.code_list["code_z"])
        self.code_list["code_y_t_by_z"] = self.EC_Y(self.image_list["y_t_by_z"])
        self.image_list["z_r_c_by_y"] = self.DC_Z(self.code_list["code_y_t_by_z"])
        self.image_list["w_t_by_z"] = self.DC_W(self.code_list["code_z"])
        self.code_list["code_w_t_by_z"] = self.EC_W(self.image_list["w_t_by_z"])
        self.image_list["z_r_c_by_w"] = self.DC_Z(self.code_list["code_w_t_by_z"])

        self.image_list["x_t_by_w"] = self.DC_X(self.code_list["code_w"])
        self.code_list["code_x_t_by_w"] = self.EC_X(self.image_list["x_t_by_w"])
        self.image_list["w_r_c_by_x"] = self.DC_W(self.code_list["code_x_t_by_w"])
        # self.image_list["y_t_by_w"] = self.DC_Y(self.code_list["code_w"])
        # self.code_list["code_y_t_by_w"] = self.EC_Y(self.image_list["y_t_by_w"])
        # self.image_list["w_r_c_by_y"] = self.DC_W(self.code_list["code_y_t_by_w"])
        self.image_list["z_t_by_w"] = self.DC_Z(self.code_list["code_w"])
        self.code_list["code_z_t_by_w"] = self.EC_Z(self.image_list["z_t_by_w"])
        self.image_list["w_r_c_by_z"] = self.DC_W(self.code_list["code_z_t_by_w"])

        self.judge_list["j_x_g"], self.judge_list["j_x_g_c"] = self.D_M(self.image_list["x_g"])
        self.judge_list["j_y_g"], self.judge_list["j_y_g_c"] = self.D_M(self.image_list["y_g"])
        self.judge_list["j_z_g"], self.judge_list["j_z_g_c"] = self.D_M(self.image_list["z_g"])
        self.judge_list["j_w_g"], self.judge_list["j_w_g_c"] = self.D_M(self.image_list["w_g"])

        self.judge_list["j_x"], self.judge_list["j_x_c"] = self.D_M(x)
        self.judge_list["j_y"], self.judge_list["j_y_c"] = self.D_M(y)
        self.judge_list["j_z"], self.judge_list["j_z_c"] = self.D_M(z)
        self.judge_list["j_w"], self.judge_list["j_w_c"] = self.D_M(w)

        self.judge_list["j_x_t_by_y"], self.judge_list["j_x_t_c_by_y"] = self.D_M(self.image_list["x_t_by_y"])
        # self.judge_list["j_x_t_by_z"], self.judge_list["j_x_t_c_by_z"] = self.D_M(self.image_list["x_t_by_z"])
        self.judge_list["j_x_t_by_w"], self.judge_list["j_x_t_c_by_w"] = self.D_M(self.image_list["x_t_by_w"])

        self.judge_list["j_y_t_by_x"], self.judge_list["j_y_t_c_by_x"] = self.D_M(self.image_list["y_t_by_x"])
        self.judge_list["j_y_t_by_z"], self.judge_list["j_y_t_c_by_z"] = self.D_M(self.image_list["y_t_by_z"])
        # self.judge_list["j_y_t_by_w"], self.judge_list["j_y_t_c_by_w"] = self.D_M(self.image_list["y_t_by_w"])

        # self.judge_list["j_z_t_by_x"], self.judge_list["j_z_t_c_by_x"] = self.D_M(self.image_list["z_t_by_x"])
        self.judge_list["j_z_t_by_y"], self.judge_list["j_z_t_c_by_y"] = self.D_M(self.image_list["z_t_by_y"])
        self.judge_list["j_z_t_by_w"], self.judge_list["j_z_t_c_by_w"] = self.D_M(self.image_list["z_t_by_w"])

        self.judge_list["j_w_t_by_x"], self.judge_list["j_w_t_c_by_x"] = self.D_M(self.image_list["w_t_by_x"])
        # self.judge_list["j_w_t_by_y"], self.judge_list["j_w_t_c_by_y"] = self.D_M(self.image_list["w_t_by_y"])
        self.judge_list["j_w_t_by_z"], self.judge_list["j_w_t_c_by_z"] = self.D_M(self.image_list["w_t_by_z"])

        self.judge_list["j_code_x"] = self.FD_R(self.code_list["code_x"])
        self.judge_list["j_code_y"] = self.FD_R(self.code_list["code_y"])
        self.judge_list["j_code_z"] = self.FD_R(self.code_list["code_z"])
        self.judge_list["j_code_w"] = self.FD_R(self.code_list["code_w"])
        self.judge_list["j_code_rm"] = self.FD_R(self.code_list["code_rm"])

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(self.judge_list["j_x"], 1.0) * 35
        D_loss += self.mse_loss(self.judge_list["j_x_g"], 0.0) * 20
        G_loss += self.mse_loss(self.judge_list["j_x_g"], 1.0) * 25
        D_loss += self.mse_loss(self.judge_list["j_x_t_by_y"], 0.0) * 10
        # D_loss += self.mse_loss(self.judge_list["j_x_t_by_z"], 0.0) * 10
        D_loss += self.mse_loss(self.judge_list["j_x_t_by_w"], 0.0) * 10
        G_loss += self.mse_loss(self.judge_list["j_x_t_by_y"], 1.0) * 15
        # G_loss += self.mse_loss(self.judge_list["j_x_t_by_z"], 1.0) * 15
        G_loss += self.mse_loss(self.judge_list["j_x_t_by_w"], 1.0) * 15

        D_loss += self.mse_loss(self.judge_list["j_y"], 1.0) * 35
        D_loss += self.mse_loss(self.judge_list["j_y_g"], 0.0) * 20
        G_loss += self.mse_loss(self.judge_list["j_y_g"], 1.0) * 25
        D_loss += self.mse_loss(self.judge_list["j_y_t_by_x"], 0.0) * 10
        D_loss += self.mse_loss(self.judge_list["j_y_t_by_z"], 0.0) * 10
        # D_loss += self.mse_loss(self.judge_list["j_y_t_by_w"], 0.0) * 10
        G_loss += self.mse_loss(self.judge_list["j_y_t_by_x"], 1.0) * 15
        G_loss += self.mse_loss(self.judge_list["j_y_t_by_z"], 1.0) * 15
        # G_loss += self.mse_loss(self.judge_list["j_y_t_by_w"], 1.0) * 15

        D_loss += self.mse_loss(self.judge_list["j_z"], 1.0) * 35
        D_loss += self.mse_loss(self.judge_list["j_z_g"], 0.0) * 20
        G_loss += self.mse_loss(self.judge_list["j_z_g"], 1.0) * 25
        # D_loss += self.mse_loss(self.judge_list["j_z_t_by_x"], 0.0) * 10
        D_loss += self.mse_loss(self.judge_list["j_z_t_by_y"], 0.0) * 10
        D_loss += self.mse_loss(self.judge_list["j_z_t_by_w"], 0.0) * 10
        # G_loss += self.mse_loss(self.judge_list["j_z_t_by_x"], 1.0) * 15
        G_loss += self.mse_loss(self.judge_list["j_z_t_by_y"], 1.0) * 15
        G_loss += self.mse_loss(self.judge_list["j_z_t_by_w"], 1.0) * 15

        D_loss += self.mse_loss(self.judge_list["j_w"], 1.0) * 35
        D_loss += self.mse_loss(self.judge_list["j_w_g"], 0.0) * 20
        G_loss += self.mse_loss(self.judge_list["j_w_g"], 1.0) * 25
        D_loss += self.mse_loss(self.judge_list["j_w_t_by_x"], 0.0) * 10
        # D_loss += self.mse_loss(self.judge_list["j_w_t_by_y"], 0.0) * 10
        D_loss += self.mse_loss(self.judge_list["j_w_t_by_z"], 0.0) * 10
        G_loss += self.mse_loss(self.judge_list["j_w_t_by_x"], 1.0) * 15
        # G_loss += self.mse_loss(self.judge_list["j_w_t_by_y"], 1.0) * 15
        G_loss += self.mse_loss(self.judge_list["j_w_t_by_z"], 1.0) * 15

        D_loss += self.mse_loss(self.judge_list["j_x_c"], cx) * 25
        D_loss += self.mse_loss(self.judge_list["j_y_c"], cy) * 25
        D_loss += self.mse_loss(self.judge_list["j_z_c"], cz) * 25
        D_loss += self.mse_loss(self.judge_list["j_w_c"], cw) * 25

        G_loss += self.mse_loss(self.judge_list["j_x_g_c"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_y_g_c"], cy) * 25
        G_loss += self.mse_loss(self.judge_list["j_z_g_c"], cz) * 25
        G_loss += self.mse_loss(self.judge_list["j_w_g_c"], cw) * 25

        G_loss += self.mse_loss(self.judge_list["j_x_t_by_y"], cx) * 25
        # G_loss += self.mse_loss(self.judge_list["j_x_t_by_z"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_x_t_by_w"], cx) * 25

        G_loss += self.mse_loss(self.judge_list["j_y_t_by_x"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_y_t_by_z"], cx) * 25
        # G_loss += self.mse_loss(self.judge_list["j_y_t_by_w"], cx) * 25

        # G_loss += self.mse_loss(self.judge_list["j_z_t_by_x"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_z_t_by_y"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_z_t_by_w"], cx) * 25

        G_loss += self.mse_loss(self.judge_list["j_w_t_by_x"], cx) * 25
        # G_loss += self.mse_loss(self.judge_list["j_w_t_by_y"], cx) * 25
        G_loss += self.mse_loss(self.judge_list["j_w_t_by_z"], cx) * 25

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_loss += self.mse_loss(self.judge_list["j_code_rm"], 0.0) * 4
        D_loss += self.mse_loss(self.judge_list["j_code_x"], 1.0)
        D_loss += self.mse_loss(self.judge_list["j_code_y"], 1.0)
        D_loss += self.mse_loss(self.judge_list["j_code_z"], 1.0)
        D_loss += self.mse_loss(self.judge_list["j_code_w"], 1.0)
        G_loss += self.mse_loss(self.judge_list["j_code_rm"], 1.0) * 4

        # 输入的结构特征图的重建自监督损失
        G_loss += self.mse_loss(self.image_list["f"], self.image_list["f_x_g_r"]) * 20
        G_loss += self.mse_loss(self.image_list["f"], self.image_list["f_y_g_r"]) * 20
        G_loss += self.mse_loss(self.image_list["f"], self.image_list["f_z_g_r"]) * 20
        G_loss += self.mse_loss(self.image_list["f"], self.image_list["f_w_g_r"]) * 20

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(self.prob_list["label_expand"][:, :, :, 0],
                                self.prob_list["l_g_prob"][:, :, :, 0]) * 0.1 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 1], self.prob_list["l_g_prob"][:, :, :, 1]) \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 2],
                                  self.prob_list["l_g_prob"][:, :, :, 2]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 3],
                                  self.prob_list["l_g_prob"][:, :, :, 3]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 4],
                                  self.prob_list["l_g_prob"][:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, self.image_list["l_g"])

        G_loss += self.mse_loss(self.prob_list["label_expand"][:, :, :, 0],
                                self.prob_list["l_g_prob_by_x"][:, :, :, 0]) * 0.1 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 1],
                                  self.prob_list["l_g_prob_by_x"][:, :, :, 1]) \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 2],
                                  self.prob_list["l_g_prob_by_x"][:, :, :, 2]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 3],
                                  self.prob_list["l_g_prob_by_x"][:, :, :, 3]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 4],
                                  self.prob_list["l_g_prob_by_x"][:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, self.image_list["l_g_by_x"])

        G_loss += self.mse_loss(self.prob_list["label_expand"][:, :, :, 0],
                                self.prob_list["l_g_prob_by_y"][:, :, :, 0]) * 0.1 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 1],
                                  self.prob_list["l_g_prob_by_y"][:, :, :, 1]) \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 2],
                                  self.prob_list["l_g_prob_by_y"][:, :, :, 2]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 3],
                                  self.prob_list["l_g_prob_by_y"][:, :, :, 3]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 4],
                                  self.prob_list["l_g_prob_by_y"][:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, self.image_list["l_g_by_y"])

        G_loss += self.mse_loss(self.prob_list["label_expand"][:, :, :, 0],
                                self.prob_list["l_g_prob_by_z"][:, :, :, 0]) * 0.1 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 1],
                                  self.prob_list["l_g_prob_by_z"][:, :, :, 1]) \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 2],
                                  self.prob_list["l_g_prob_by_z"][:, :, :, 2]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 3],
                                  self.prob_list["l_g_prob_by_z"][:, :, :, 3]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 4],
                                  self.prob_list["l_g_prob_by_z"][:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, self.image_list["l_g_by_z"])

        G_loss += self.mse_loss(self.prob_list["label_expand"][:, :, :, 0],
                                self.prob_list["l_g_prob_by_w"][:, :, :, 0]) * 0.1 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 1],
                                  self.prob_list["l_g_prob_by_w"][:, :, :, 1]) \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 2],
                                  self.prob_list["l_g_prob_by_w"][:, :, :, 2]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 3],
                                  self.prob_list["l_g_prob_by_w"][:, :, :, 3]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand"][:, :, :, 4],
                                  self.prob_list["l_g_prob_by_w"][:, :, :, 4]) * 5
        G_loss += self.mse_loss(l, self.image_list["l_g_by_w"])

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(self.prob_list["label_expand_x"][:, :, :, 0],
                                self.prob_list["l_f_prob_by_x"][:, :, :, 0]) \
                  + self.mse_loss(self.prob_list["label_expand_x"][:, :, :, 1],
                                  self.prob_list["l_f_prob_by_x"][:, :, :, 1]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand_x"][:, :, :, 2],
                                  self.prob_list["l_f_prob_by_x"][:, :, :, 2]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_x"][:, :, :, 3],
                                  self.prob_list["l_f_prob_by_x"][:, :, :, 3]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_x"][:, :, :, 4],
                                  self.prob_list["l_f_prob_by_x"][:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_x, self.image_list["l_f_by_x"]) * 5

        G_loss += self.mse_loss(self.prob_list["label_expand_y"][:, :, :, 0],
                                self.prob_list["l_f_prob_by_y"][:, :, :, 0]) \
                  + self.mse_loss(self.prob_list["label_expand_y"][:, :, :, 1],
                                  self.prob_list["l_f_prob_by_y"][:, :, :, 1]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand_y"][:, :, :, 2],
                                  self.prob_list["l_f_prob_by_y"][:, :, :, 2]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_y"][:, :, :, 3],
                                  self.prob_list["l_f_prob_by_y"][:, :, :, 3]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_y"][:, :, :, 4],
                                  self.prob_list["l_f_prob_by_y"][:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_y, self.image_list["l_f_by_y"]) * 5

        G_loss += self.mse_loss(self.prob_list["label_expand_z"][:, :, :, 0],
                                self.prob_list["l_f_prob_by_z"][:, :, :, 0]) \
                  + self.mse_loss(self.prob_list["label_expand_z"][:, :, :, 1],
                                  self.prob_list["l_f_prob_by_z"][:, :, :, 1]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand_z"][:, :, :, 2],
                                  self.prob_list["l_f_prob_by_z"][:, :, :, 2]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_z"][:, :, :, 3],
                                  self.prob_list["l_f_prob_by_z"][:, :, :, 3]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_z"][:, :, :, 4],
                                  self.prob_list["l_f_prob_by_z"][:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_z, self.image_list["l_f_by_z"]) * 5

        G_loss += self.mse_loss(self.prob_list["label_expand_w"][:, :, :, 0],
                                self.prob_list["l_f_prob_by_w"][:, :, :, 0]) \
                  + self.mse_loss(self.prob_list["label_expand_w"][:, :, :, 1],
                                  self.prob_list["l_f_prob_by_w"][:, :, :, 1]) * 5 \
                  + self.mse_loss(self.prob_list["label_expand_w"][:, :, :, 2],
                                  self.prob_list["l_f_prob_by_w"][:, :, :, 2]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_w"][:, :, :, 3],
                                  self.prob_list["l_f_prob_by_w"][:, :, :, 3]) * 15 \
                  + self.mse_loss(self.prob_list["label_expand_w"][:, :, :, 4],
                                  self.prob_list["l_f_prob_by_w"][:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_w, self.image_list["l_f_by_w"]) * 5

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(self.image_list["x_g"], self.image_list["x_g_t_by_y"]) * 2
        # G_loss += self.mse_loss(self.image_list["x_g"], self.image_list["x_g_t_by_z"]) * 2
        G_loss += self.mse_loss(self.image_list["x_g"], self.image_list["x_g_t_by_w"]) * 2

        G_loss += self.mse_loss(self.image_list["y_g"], self.image_list["y_g_t_by_x"]) * 2
        G_loss += self.mse_loss(self.image_list["y_g"], self.image_list["y_g_t_by_z"]) * 2
        # G_loss += self.mse_loss(self.image_list["y_g"], self.image_list["y_g_t_by_w"]) * 2

        # G_loss += self.mse_loss(self.image_list["z_g"], self.image_list["z_g_t_by_x"]) * 2
        G_loss += self.mse_loss(self.image_list["z_g"], self.image_list["z_g_t_by_y"]) * 2
        G_loss += self.mse_loss(self.image_list["z_g"], self.image_list["z_g_t_by_w"]) * 2

        G_loss += self.mse_loss(self.image_list["w_g"], self.image_list["w_g_t_by_x"]) * 2
        # G_loss += self.mse_loss(self.image_list["w_g"], self.image_list["w_g_t_by_y"]) * 2
        G_loss += self.mse_loss(self.image_list["w_g"], self.image_list["w_g_t_by_z"]) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, self.image_list["x_g"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["y_g"] * self.image_list["mask"]) * 1.5

        G_loss += self.mse_loss(0.0, self.image_list["x_g_t_by_y"] * self.image_list["mask"]) * 1.5
        # G_loss += self.mse_loss(0.0, self.image_list["x_g_t_by_z"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["x_g_t_by_w"] * self.image_list["mask"]) * 1.5

        G_loss += self.mse_loss(0.0, self.image_list["y_g_t_by_x"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["y_g_t_by_z"] * self.image_list["mask"]) * 1.5
        # G_loss += self.mse_loss(0.0, self.image_list["y_g_t_by_w"] * self.image_list["mask"]) * 1.5

        # G_loss += self.mse_loss(0.0, self.image_list["z_g_t_by_x"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["z_g_t_by_y"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["z_g_t_by_w"] * self.image_list["mask"]) * 1.5

        G_loss += self.mse_loss(0.0, self.image_list["w_g_t_by_x"] * self.image_list["mask"]) * 1.5
        # G_loss += self.mse_loss(0.0, self.image_list["w_g_t_by_y"] * self.image_list["mask"]) * 1.5
        G_loss += self.mse_loss(0.0, self.image_list["w_g_t_by_z"] * self.image_list["mask"]) * 1.5

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, self.image_list["x_r"]) * 5
        G_loss += self.mse_loss(y, self.image_list["y_r"]) * 5
        G_loss += self.mse_loss(z, self.image_list["z_r"]) * 5
        G_loss += self.mse_loss(w, self.image_list["w_r"]) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, self.image_list["x_r_c_by_y"]) * 10
        # G_loss += self.mse_loss(x, self.image_list["x_r_c_by_z"]) * 10
        G_loss += self.mse_loss(x, self.image_list["x_r_c_by_w"]) * 10

        G_loss += self.mse_loss(y, self.image_list["y_r_c_by_x"]) * 10
        G_loss += self.mse_loss(y, self.image_list["y_r_c_by_z"]) * 10
        # G_loss += self.mse_loss(y, self.image_list["y_r_c_by_w"]) * 10

        # G_loss += self.mse_loss(z, self.image_list["z_r_c_by_x"]) * 10
        G_loss += self.mse_loss(z, self.image_list["z_r_c_by_y"]) * 10
        G_loss += self.mse_loss(z, self.image_list["z_r_c_by_w"]) * 10

        G_loss += self.mse_loss(w, self.image_list["w_r_c_by_x"]) * 10
        # G_loss += self.mse_loss(w, self.image_list["w_r_c_by_y"]) * 10
        G_loss += self.mse_loss(w, self.image_list["w_r_c_by_z"]) * 10

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, self.image_list["x_t_by_y"] * self.image_list["mask_x"]) * 2
        # G_loss += self.mse_loss(0.0, self.image_list["x_t_by_z"] * self.image_list["mask_x"]) * 2
        G_loss += self.mse_loss(0.0, self.image_list["x_t_by_w"] * self.image_list["mask_x"]) * 2

        G_loss += self.mse_loss(0.0, self.image_list["y_t_by_x"] * self.image_list["mask_x"]) * 2
        G_loss += self.mse_loss(0.0, self.image_list["y_t_by_z"] * self.image_list["mask_x"]) * 2
        # G_loss += self.mse_loss(0.0, self.image_list["y_t_by_w"] * self.image_list["mask_x"]) * 2

        # G_loss += self.mse_loss(0.0, self.image_list["z_t_by_x"] * self.image_list["mask_x"]) * 2
        G_loss += self.mse_loss(0.0, self.image_list["z_t_by_y"] * self.image_list["mask_x"]) * 2
        G_loss += self.mse_loss(0.0, self.image_list["z_t_by_w"] * self.image_list["mask_x"]) * 2

        G_loss += self.mse_loss(0.0, self.image_list["w_t_by_x"] * self.image_list["mask_x"]) * 2
        # G_loss += self.mse_loss(0.0, self.image_list["w_t_by_y"] * self.image_list["mask_x"]) * 2
        G_loss += self.mse_loss(0.0, self.image_list["w_t_by_z"] * self.image_list["mask_x"]) * 2

        G_loss += self.mse_loss(0.0, self.image_list["x_r"] * self.image_list["mask_x"]) * 0.5
        G_loss += self.mse_loss(0.0, self.image_list["y_r"] * self.image_list["mask_y"]) * 0.5
        G_loss += self.mse_loss(0.0, self.image_list["z_r"] * self.image_list["mask_z"]) * 0.5
        G_loss += self.mse_loss(0.0, self.image_list["w_r"] * self.image_list["mask_w"]) * 0.5

        # 通过解码器生成X模态与Y模态图的编码与X模态与Y模态图经过编码器得到的编码的自监督语义一致性损失
        G_loss += self.mse_loss(self.code_list["code_rm"], self.code_list["code_x_g"])
        G_loss += self.mse_loss(self.code_list["code_rm"], self.code_list["code_y_g"])
        G_loss += self.mse_loss(self.code_list["code_rm"], self.code_list["code_z_g"])
        G_loss += self.mse_loss(self.code_list["code_rm"], self.code_list["code_w_g"])

        G_loss += self.mse_loss(self.code_list["code_x_g"], self.code_list["code_y_g"]) * 0.5
        G_loss += self.mse_loss(self.code_list["code_x_g"], self.code_list["code_z_g"]) * 0.5
        G_loss += self.mse_loss(self.code_list["code_x_g"], self.code_list["code_w_g"]) * 0.5
        G_loss += self.mse_loss(self.code_list["code_y_g"], self.code_list["code_z_g"]) * 0.5
        G_loss += self.mse_loss(self.code_list["code_y_g"], self.code_list["code_w_g"]) * 0.5
        G_loss += self.mse_loss(self.code_list["code_z_g"], self.code_list["code_w_g"]) * 0.5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(self.code_list["code_x"], self.code_list["code_y_t_by_x"]) * 5
        # G_loss += self.mse_loss(self.code_list["code_x"], self.code_list["code_z_t_by_x"]) * 5
        G_loss += self.mse_loss(self.code_list["code_x"], self.code_list["code_w_t_by_x"]) * 5

        G_loss += self.mse_loss(self.code_list["code_y"], self.code_list["code_x_t_by_y"]) * 5
        G_loss += self.mse_loss(self.code_list["code_y"], self.code_list["code_z_t_by_y"]) * 5
        # G_loss += self.mse_loss(self.code_list["code_y"], self.code_list["code_w_t_by_y"]) * 5

        # G_loss += self.mse_loss(self.code_list["code_z"], self.code_list["code_x_t_by_z"]) * 5
        G_loss += self.mse_loss(self.code_list["code_z"], self.code_list["code_y_t_by_z"]) * 5
        G_loss += self.mse_loss(self.code_list["code_z"], self.code_list["code_w_t_by_z"]) * 5

        G_loss += self.mse_loss(self.code_list["code_w"], self.code_list["code_x_t_by_w"]) * 5
        # G_loss += self.mse_loss(self.code_list["code_w"], self.code_list["code_y_t_by_w"]) * 5
        G_loss += self.mse_loss(self.code_list["code_w"], self.code_list["code_z_t_by_w"]) * 5

        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.DC_L.variables
                + self.EC_X.variables
                + self.DC_X.variables
                + self.EC_Y.variables
                + self.DC_Y.variables
                + self.EC_Z.variables
                + self.DC_Z.variables
                + self.EC_W.variables
                + self.DC_W.variables
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

    def histogram_summary(self, j_list):
        pass
        # judge_list_XY, judge_list_XW, judge_list_YZ, judge_list_ZW = \
        #     j_list[0], j_list[1], j_list[2], j_list[3]
        #
        # xy_j_x_g, xy_j_y_g, xy_j_code_rm, xy_j_code_x, xy_j_code_y = \
        #     judge_list_XY[0], judge_list_XY[1], judge_list_XY[2], judge_list_XY[5], judge_list_XY[6]
        # xw_j_x_g, xw_j_y_g, xw_j_code_rm, xw_j_code_x, xw_j_code_y = \
        #     judge_list_XW[0], judge_list_XW[1], judge_list_XW[2], judge_list_XW[5], judge_list_XW[6]
        # yz_j_x_g, yz_j_y_g, yz_j_code_rm, yz_j_code_x, yz_j_code_y = \
        #     judge_list_YZ[0], judge_list_YZ[1], judge_list_YZ[2], judge_list_YZ[5], judge_list_YZ[6]
        # zw_j_x_g, zw_j_y_g, zw_j_code_rm, zw_j_code_x, zw_j_code_y = \
        #     judge_list_ZW[0], judge_list_ZW[1], judge_list_ZW[2], judge_list_ZW[5], judge_list_ZW[6]
        #
        # tf.summary.histogram('discriminator/TRUE/XY/j_code_x', xy_j_code_x)
        # tf.summary.histogram('discriminator/TRUE/XY/j_code_y', xy_j_code_y)
        # tf.summary.histogram('discriminator/FALSE/XY/j_x_g', xy_j_x_g)
        # tf.summary.histogram('discriminator/FALSE/XY/j_y_g', xy_j_y_g)
        # tf.summary.histogram('discriminator/FALSE/XY/j_code_rm', xy_j_code_rm)
        #
        # tf.summary.histogram('discriminator/TRUE/XW/j_code_x', xw_j_code_x)
        # tf.summary.histogram('discriminator/TRUE/XW/j_code_y', xw_j_code_y)
        # tf.summary.histogram('discriminator/FALSE/XW/j_x_g', xw_j_x_g)
        # tf.summary.histogram('discriminator/FALSE/XW/j_y_g', xw_j_y_g)
        # tf.summary.histogram('discriminator/FALSE/XW/j_code_rm', xw_j_code_rm)
        #
        # tf.summary.histogram('discriminator/TRUE/YZ/j_code_x', yz_j_code_x)
        # tf.summary.histogram('discriminator/TRUE/YZ/j_code_y', yz_j_code_y)
        # tf.summary.histogram('discriminator/FALSE/YZ/j_x_g', yz_j_x_g)
        # tf.summary.histogram('discriminator/FALSE/YZ/j_y_g', yz_j_y_g)
        # tf.summary.histogram('discriminator/FALSE/YZ/j_code_rm', yz_j_code_rm)
        #
        # tf.summary.histogram('discriminator/TRUE/ZW/j_code_x', zw_j_code_x)
        # tf.summary.histogram('discriminator/TRUE/ZW/j_code_y', zw_j_code_y)
        # tf.summary.histogram('discriminator/FALSE/ZW/j_x_g', zw_j_x_g)
        # tf.summary.histogram('discriminator/FALSE/ZW/j_y_g', zw_j_y_g)
        # tf.summary.histogram('discriminator/FALSE/ZW/j_code_rm', zw_j_code_rm)

    def loss_summary(self, loss_list):
        pass
        # G_loss, D_loss = loss_list[0], loss_list[1]
        # tf.summary.scalar('loss/G_loss', G_loss)
        # tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_list):
        return []
        # code_list_XY, code_list_XW, code_list_YZ, code_list_ZW = \
        #     code_list[0], code_list[1], code_list[2], code_list[3]
        #
        # xy_code_rm, xy_code_x_g, xy_code_y_g = code_list_XY[0], code_list_XY[1], code_list_XY[3]
        # xw_code_rm, xw_code_x_g, xw_code_y_g = code_list_XW[0], code_list_XW[1], code_list_XW[3]
        # yz_code_rm, yz_code_x_g, yz_code_y_g = code_list_YZ[0], code_list_YZ[1], code_list_YZ[3]
        # zw_code_rm, zw_code_x_g, zw_code_y_g = code_list_ZW[0], code_list_ZW[1], code_list_ZW[3]
        #
        # list = [self.PSNR(xy_code_rm, xy_code_x_g), self.PSNR(xy_code_rm, xy_code_y_g),
        #         self.PSNR(xy_code_x_g, xy_code_y_g),
        #         self.PSNR(xw_code_rm, xw_code_x_g), self.PSNR(xw_code_rm, xw_code_y_g),
        #         self.PSNR(xw_code_x_g, xw_code_y_g),
        #         self.PSNR(yz_code_rm, yz_code_x_g), self.PSNR(yz_code_rm, yz_code_y_g),
        #         self.PSNR(yz_code_x_g, yz_code_y_g),
        #         self.PSNR(zw_code_rm, zw_code_x_g), self.PSNR(zw_code_rm, zw_code_y_g),
        #         self.PSNR(zw_code_x_g, zw_code_y_g),
        #
        #         self.SSIM(xy_code_rm, xy_code_x_g), self.SSIM(xy_code_rm, xy_code_y_g),
        #         self.SSIM(xy_code_x_g, xy_code_y_g),
        #         self.SSIM(xw_code_rm, xw_code_x_g), self.SSIM(xw_code_rm, xw_code_y_g),
        #         self.SSIM(xw_code_x_g, xw_code_y_g),
        #         self.SSIM(yz_code_rm, yz_code_x_g), self.SSIM(yz_code_rm, yz_code_y_g),
        #         self.SSIM(yz_code_x_g, yz_code_y_g),
        #         self.SSIM(zw_code_rm, zw_code_x_g), self.SSIM(zw_code_rm, zw_code_y_g),
        #         self.SSIM(zw_code_x_g, zw_code_y_g)]
        # return list

    def evaluation_code_summary(self, evluation_list):
        pass
        # tf.summary.scalar('evaluation_code/PSNR/XY/code_rm__VS__code_x_g', evluation_list[0])
        # tf.summary.scalar('evaluation_code/PSNR/XY/code_rm__VS__code_y_g', evluation_list[1])
        # tf.summary.scalar('evaluation_code/PSNR/XY/code_x_g__VS__code_y_g', evluation_list[2])
        # tf.summary.scalar('evaluation_code/PSNR/XW/code_rm__VS__code_x_g', evluation_list[3])
        # tf.summary.scalar('evaluation_code/PSNR/XW/code_rm__VS__code_y_g', evluation_list[4])
        # tf.summary.scalar('evaluation_code/PSNR/XW/code_x_g__VS__code_y_g', evluation_list[5])
        # tf.summary.scalar('evaluation_code/PSNR/YZ/code_rm__VS__code_x_g', evluation_list[6])
        # tf.summary.scalar('evaluation_code/PSNR/YZ/code_rm__VS__code_y_g', evluation_list[7])
        # tf.summary.scalar('evaluation_code/PSNR/YZ/code_x_g__VS__code_y_g', evluation_list[8])
        # tf.summary.scalar('evaluation_code/PSNR/ZW/code_rm__VS__code_x_g', evluation_list[9])
        # tf.summary.scalar('evaluation_code/PSNR/ZW/code_rm__VS__code_y_g', evluation_list[10])
        # tf.summary.scalar('evaluation_code/PSNR/ZW/code_x_g__VS__code_y_g', evluation_list[11])
        #
        # tf.summary.scalar('evaluation_code/SSIM/XY/code_rm__VS__code_x_g', evluation_list[12])
        # tf.summary.scalar('evaluation_code/SSIM/XY/code_rm__VS__code_y_g', evluation_list[13])
        # tf.summary.scalar('evaluation_code/SSIM/XY/code_x_g__VS__code_y_g', evluation_list[14])
        # tf.summary.scalar('evaluation_code/SSIM/XW/code_rm__VS__code_x_g', evluation_list[15])
        # tf.summary.scalar('evaluation_code/SSIM/XW/code_rm__VS__code_y_g', evluation_list[16])
        # tf.summary.scalar('evaluation_code/SSIM/XW/code_x_g__VS__code_y_g', evluation_list[17])
        # tf.summary.scalar('evaluation_code/SSIM/YZ/code_rm__VS__code_x_g', evluation_list[18])
        # tf.summary.scalar('evaluation_code/SSIM/YZ/code_rm__VS__code_y_g', evluation_list[19])
        # tf.summary.scalar('evaluation_code/SSIM/YZ/code_x_g__VS__code_y_g', evluation_list[20])
        # tf.summary.scalar('evaluation_code/SSIM/ZW/code_rm__VS__code_x_g', evluation_list[21])
        # tf.summary.scalar('evaluation_code/SSIM/ZW/code_rm__VS__code_y_g', evluation_list[22])
        # tf.summary.scalar('evaluation_code/SSIM/ZW/code_x_g__VS__code_y_g', evluation_list[23])

    def evaluation(self, image_list):
        return []
        # image_list_XY, image_list_XW, image_list_YZ, image_list_ZW = \
        #     image_list[0], image_list[1], image_list[2], image_list[3]
        #
        # xy_l, xy_l_g, xy_x, xy_y, xy_x_r, xy_y_r = \
        #     image_list_XY[0], image_list_XY[6], image_list_XY[11], image_list_XY[12], image_list_XY[13], image_list_XY[
        #         14]
        # xw_l, xw_l_g, xw_x, xw_y, xw_x_r, xw_y_r = \
        #     image_list_XW[0], image_list_XW[6], image_list_XW[11], image_list_XW[12], image_list_XW[13], image_list_XW[
        #         14]
        # yz_l, yz_l_g, yz_x, yz_y, yz_x_r, yz_y_r = \
        #     image_list_YZ[0], image_list_YZ[6], image_list_YZ[11], image_list_YZ[12], image_list_YZ[13], image_list_YZ[
        #         14]
        # zw_l, zw_l_g, zw_x, zw_y, zw_x_r, zw_y_r = \
        #     image_list_ZW[0], image_list_ZW[6], image_list_ZW[11], image_list_ZW[12], image_list_ZW[13], image_list_ZW[
        #         14]
        # list = [self.PSNR(xy_x, xy_x_r), self.PSNR(xy_y, xy_y_r),
        #         self.PSNR(xy_l, xy_l_g),
        #
        #         self.PSNR(xw_x, xw_x_r), self.PSNR(xw_y, xw_y_r),
        #         self.PSNR(xw_l, xw_l_g),
        #
        #         self.PSNR(yz_x, yz_x_r), self.PSNR(yz_y, yz_y_r),
        #         self.PSNR(yz_l, yz_l_g),
        #
        #         self.PSNR(zw_x, zw_x_r), self.PSNR(zw_y, zw_y_r),
        #         self.PSNR(zw_l, zw_l_g),
        #
        #         self.SSIM(xy_x, xy_x_r), self.SSIM(xy_y, xy_y_r),
        #         self.SSIM(xy_l, xy_l_g),
        #
        #         self.SSIM(xw_x, xw_x_r), self.SSIM(xw_y, xw_y_r),
        #         self.SSIM(xw_l, xw_l_g),
        #
        #         self.SSIM(yz_x, yz_x_r), self.SSIM(yz_y, yz_y_r),
        #         self.SSIM(yz_l, yz_l_g),
        #
        #         self.SSIM(zw_x, zw_x_r), self.SSIM(zw_y, zw_y_r),
        #         self.SSIM(zw_l, zw_l_g)
        #         ]
        # return list
    def evaluation_summary(self, evluation_list):
        pass
        # tf.summary.scalar('evaluation/PSNR/XY/x__VS__x_r', evluation_list[0])
        # tf.summary.scalar('evaluation/PSNR/XY/y__VS__y_r', evluation_list[1])
        # tf.summary.scalar('evaluation/PSNR/XY/l_input__VS__l_g', evluation_list[2])
        # tf.summary.scalar('evaluation/PSNR/XW/x__VS__x_r', evluation_list[3])
        # tf.summary.scalar('evaluation/PSNR/XW/y__VS__y_r', evluation_list[4])
        # tf.summary.scalar('evaluation/PSNR/XW/l_input__VS__l_g', evluation_list[5])
        # tf.summary.scalar('evaluation/PSNR/YZ/x__VS__x_r', evluation_list[6])
        # tf.summary.scalar('evaluation/PSNR/YZ/y__VS__y_r', evluation_list[7])
        # tf.summary.scalar('evaluation/PSNR/YZ/l_input__VS__l_g', evluation_list[8])
        # tf.summary.scalar('evaluation/PSNR/ZW/x__VS__x_r', evluation_list[9])
        # tf.summary.scalar('evaluation/PSNR/ZW/y__VS__y_r', evluation_list[10])
        # tf.summary.scalar('evaluation/PSNR/ZW/l_input__VS__l_g', evluation_list[11])
        #
        # tf.summary.scalar('evaluation/SSIM/XY/x__VS__x_r', evluation_list[12])
        # tf.summary.scalar('evaluation/SSIM/XY/y__VS__y_r', evluation_list[13])
        # tf.summary.scalar('evaluation/SSIM/XY/l_input__VS__l_g', evluation_list[14])
        # tf.summary.scalar('evaluation/SSIM/XW/x__VS__x_r', evluation_list[15])
        # tf.summary.scalar('evaluation/SSIM/XW/y__VS__y_r', evluation_list[16])
        # tf.summary.scalar('evaluation/SSIM/XW/l_input__VS__l_g', evluation_list[17])
        # tf.summary.scalar('evaluation/SSIM/YZ/x__VS__x_r', evluation_list[18])
        # tf.summary.scalar('evaluation/SSIM/YZ/y__VS__y_r', evluation_list[19])
        # tf.summary.scalar('evaluation/SSIM/YZ/l_input__VS__l_g', evluation_list[20])
        # tf.summary.scalar('evaluation/SSIM/ZW/x__VS__x_r', evluation_list[21])
        # tf.summary.scalar('evaluation/SSIM/ZW/y__VS__y_r', evluation_list[22])
        # tf.summary.scalar('evaluation/SSIM/ZW/l_input__VS__l_g', evluation_list[23])

    def image_summary(self, image_dirct):
        for key in image_dirct:
            tf.summary.image('image/'+key, image_dirct[key])
        # image_list_XY, image_list_XW, image_list_YZ, image_list_ZW = \
        #     image_list[0], image_list[1], image_list[2], image_list[3]
        #
        # xy_l, xy_f, xy_x_g, xy_y_g, xy_l_g, xy_l_g_by_x, xy_l_g_by_y, xy_f_x_g_r, xy_f_y_g_r, x, y, xy_x_r, xy_y_r = \
        #     image_list_XY[0], image_list_XY[1], image_list_XY[2], image_list_XY[3], image_list_XY[6], image_list_XY[7], \
        #     image_list_XY[8], image_list_XY[9], image_list_XY[10], image_list_XY[11], image_list_XY[12], image_list_XY[
        #         13], image_list_XY[14]
        # xw_l, xw_f, xw_x_g, xw_y_g, xw_l_g, xw_l_g_by_x, xw_l_g_by_y, xw_f_x_g_r, xw_f_y_g_r, xw_x_r, xw_y_r = \
        #     image_list_XW[0], image_list_XW[1], image_list_XW[2], image_list_XW[3], image_list_XW[6], image_list_XW[7], \
        #     image_list_XW[8], image_list_XW[9], image_list_XW[10], image_list_XW[13], image_list_XW[14]
        # yz_l, yz_f, yz_x_g, yz_y_g, yz_l_g, yz_l_g_by_x, yz_l_g_by_y, yz_f_x_g_r, yz_f_y_g_r, yz_x_r, yz_y_r = \
        #     image_list_YZ[0], image_list_YZ[1], image_list_YZ[2], image_list_YZ[3], image_list_YZ[6], image_list_YZ[7], \
        #     image_list_YZ[8], image_list_YZ[9], image_list_YZ[10], image_list_YZ[13], image_list_YZ[14]
        # zw_l, zw_f, zw_x_g, zw_y_g, zw_l_g, zw_l_g_by_x, zw_l_g_by_y, zw_f_x_g_r, zw_f_y_g_r, z, w, zw_x_r, zw_y_r = \
        #     image_list_ZW[0], image_list_ZW[1], image_list_ZW[2], image_list_ZW[3], image_list_ZW[6], image_list_ZW[7], \
        #     image_list_ZW[8], image_list_ZW[9], image_list_ZW[10], image_list_ZW[11], image_list_ZW[12], image_list_ZW[
        #         13], image_list_ZW[14]
        #
        # tf.summary.image('image/x', x)
        # tf.summary.image('image/y', y)
        # tf.summary.image('image/z', z)
        # tf.summary.image('image/w', w)
        #
        # tf.summary.image('image/XY_x_g', xy_x_g)
        # tf.summary.image('image/XY_x_r', xy_x_r)
        # tf.summary.image('image/XY_y_g', xy_y_g)
        # tf.summary.image('image/XY_y_r', xy_y_r)
        # tf.summary.image('image/XY_l_input', xy_l)
        # tf.summary.image('image/XY_l_g', xy_l_g)
        # tf.summary.image('image/XY_l_g_by_x', xy_l_g_by_x)
        # tf.summary.image('image/XY_l_g_by_y', xy_l_g_by_y)
        # tf.summary.image('image/XY_f', xy_f)
        # tf.summary.image('image/XY_f_x_g_r', xy_f_x_g_r)
        # tf.summary.image('image/XY_f_y_g_r', xy_f_y_g_r)
        #
        # tf.summary.image('image/XW_x_g', xw_x_g)
        # tf.summary.image('image/XW_x_r', xw_x_r)
        # tf.summary.image('image/XW_y_g', xw_y_g)
        # tf.summary.image('image/XW_y_r', xw_y_r)
        # tf.summary.image('image/XW_l_input', xw_l)
        # tf.summary.image('image/XW_l_g', xw_l_g)
        # tf.summary.image('image/XW_l_g_by_x', xw_l_g_by_x)
        # tf.summary.image('image/XW_l_g_by_y', xw_l_g_by_y)
        # tf.summary.image('image/XW_f', xw_f)
        # tf.summary.image('image/XW_f_x_g_r', xw_f_x_g_r)
        # tf.summary.image('image/XW_f_y_g_r', xw_f_y_g_r)
        #
        # tf.summary.image('image/YZ_x_g', yz_x_g)
        # tf.summary.image('image/YZ_x_r', yz_x_r)
        # tf.summary.image('image/YZ_y_g', yz_y_g)
        # tf.summary.image('image/YZ_y_r', yz_y_r)
        # tf.summary.image('image/YZ_l_input', yz_l)
        # tf.summary.image('image/YZ_l_g', yz_l_g)
        # tf.summary.image('image/YZ_l_g_by_x', yz_l_g_by_x)
        # tf.summary.image('image/YZ_l_g_by_y', yz_l_g_by_y)
        # tf.summary.image('image/YZ_f', yz_f)
        # tf.summary.image('image/YZ_f_x_g_r', yz_f_x_g_r)
        # tf.summary.image('image/YZ_f_y_g_r', yz_f_y_g_r)
        #
        # tf.summary.image('image/ZW_x_g', zw_x_g)
        # tf.summary.image('image/ZW_x_r', zw_x_r)
        # tf.summary.image('image/ZW_y_g', zw_y_g)
        # tf.summary.image('image/ZW_y_r', zw_y_r)
        # tf.summary.image('image/ZW_l_input', zw_l)
        # tf.summary.image('image/ZW_l_g', zw_l_g)
        # tf.summary.image('image/ZW_l_g_by_x', zw_l_g_by_x)
        # tf.summary.image('image/ZW_l_g_by_y', zw_l_g_by_y)
        # tf.summary.image('image/ZW_f', zw_f)
        # tf.summary.image('image/ZW_f_x_g_r', zw_f_x_g_r)
        # tf.summary.image('image/ZW_f_y_g_r', zw_f_y_g_r)

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
