# _*_ coding:utf-8 _*_
import tensorflow as tf
from detect_discriminator import Discriminator
from encoder import Encoder
from decoder import Decoder
from feature_discriminator import FeatureDiscriminator


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
        self.mri2mri_code_list = {}
        self.ct2ct_code_list = {}
        self.mri2ct_code_list = {}
        self.ct2mri_code_list = {}

        self.mri2mri_prob_list = {}
        self.ct2ct_prob_list = {}
        self.mri2ct_prob_list = {}
        self.ct2mri_prob_list = {}

        self.mri2mri_image_list = {}
        self.ct2ct_image_list = {}
        self.mri2ct_image_list = {}
        self.ct2mri_image_list = {}

        self.mri2mri_judge_list = {}
        self.ct2ct_judge_list = {}
        self.mri2ct_judge_list = {}
        self.ct2mri_judge_list = {}

        self.tenaor_name = {}
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=5)
        self.EC_CT = Encoder('EC_CT', ngf=ngf)
        self.DC_CT = Decoder('DC_CT', ngf=ngf)
        self.EC_MRI = Encoder('EC_MRI', ngf=ngf)
        self.DC_MRI = Decoder('DC_MRI', ngf=ngf)
        self.D_M = Discriminator('D_M', ngf=ngf)
        self.FD_M = FeatureDiscriminator('FD_M', ngf=ngf)

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

    def CT_2_CT_model(self, l_x, x, c_list):
        c_ct = 0.0
        c_mri = 1.0
        cx = c_list[0]
        cy = c_list[1]
        cz = c_list[2]
        cw = c_list[3]
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)
        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        l_x = l_x * 0.25

        mask_x = self.get_mask(x)

        code_x = self.EC_CT(x)

        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        x_r = self.DC_CT(tf.concat([code_x, cx_code], axis=-1))

        y_t_by_x = self.DC_CT(tf.concat([code_x, cy_code], axis=-1))
        code_y_t_by_x = self.EC_CT(y_t_by_x)
        x_r_c_by_y = self.DC_CT(tf.concat([code_y_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_y = self.DC_L(code_y_t_by_x)
        l_x_r_c_by_y = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        z_t_by_x = self.DC_CT(tf.concat([code_x, cz_code], axis=-1))
        code_z_t_by_x = self.EC_CT(z_t_by_x)
        x_r_c_by_z = self.DC_CT(tf.concat([code_z_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_z = self.DC_L(code_z_t_by_x)
        l_x_r_c_by_z = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        w_t_by_x = self.DC_CT(tf.concat([code_x, cw_code], axis=-1))
        code_w_t_by_x = self.EC_CT(w_t_by_x)
        x_r_c_by_w = self.DC_CT(tf.concat([code_w_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_w = self.DC_L(code_w_t_by_x)
        l_x_r_c_by_w = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        j_x, j_x_c, j_x_ct_or_mri = self.D_M(x)
        j_y_t_by_x, j_y_t_c_by_x, j_y_t_ct_or_mri = self.D_M(y_t_by_x)
        j_z_t_by_x, j_z_t_c_by_x, j_z_t_ct_or_mri = self.D_M(z_t_by_x)
        j_w_t_by_x, j_w_t_c_by_x, j_w_t_ct_or_mri = self.D_M(w_t_by_x)
        j_code_x_ct_or_mri = self.FD_M(code_x)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 45

        D_loss += self.mse_loss(j_y_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_y_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_z_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_z_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_w_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_w_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_x_c, cx) * 50
        G_loss += self.mse_loss(j_y_t_c_by_x, cy) * 50
        G_loss += self.mse_loss(j_z_t_c_by_x, cz) * 50
        G_loss += self.mse_loss(j_w_t_c_by_x, cw) * 50

        D_loss += self.mse_loss(j_x_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_y_t_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_z_t_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_w_t_ct_or_mri, c_ct) * 50

        D_loss += self.mse_loss(j_code_x_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_code_x_ct_or_mri, c_mri) * 50

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
        G_loss += self.mse_loss(l_x, l_x_r_c_by_y) * 5
        G_loss += self.mse_loss(l_x, l_x_r_c_by_z) * 5
        G_loss += self.mse_loss(l_x, l_x_r_c_by_w) * 5
        G_loss += self.mse_loss(l_x_r_c_by_y, l_x_r_c_by_z)
        G_loss += self.mse_loss(l_x_r_c_by_y, l_x_r_c_by_w)
        G_loss += self.mse_loss(l_x_r_c_by_z, l_x_r_c_by_w)

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_r_c_by_y) * 10
        G_loss += self.mse_loss(x, x_r_c_by_z) * 10
        G_loss += self.mse_loss(x, x_r_c_by_w) * 10
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_z) * 2
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_w) * 2
        G_loss += self.mse_loss(x_r_c_by_z, x_r_c_by_w) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, y_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, z_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, w_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, x_r * mask_x) * 0.5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_z_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_w_t_by_x) * 5
        G_loss += self.mse_loss(code_y_t_by_x, code_z_t_by_x)
        G_loss += self.mse_loss(code_y_t_by_x, code_w_t_by_x)
        G_loss += self.mse_loss(code_z_t_by_x, code_w_t_by_x)

        self.ct2ct_image_list["l_x"] = l_x
        self.ct2ct_image_list["x"] = x

        self.ct2ct_prob_list["label_expand_x"] = label_expand_x

        self.ct2ct_image_list["mask_x"] = mask_x

        self.ct2ct_code_list["code_x"] = code_x

        self.ct2ct_prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.ct2ct_image_list["l_f_by_x"] = l_f_by_x

        self.ct2ct_image_list["x_r"] = x_r

        self.ct2ct_image_list["y_t_by_x"] = y_t_by_x
        self.ct2ct_code_list["code_y_t_by_x"] = code_y_t_by_x
        self.ct2ct_image_list["x_r_c_by_y"] = x_r_c_by_y
        self.ct2ct_image_list["z_t_by_x"] = z_t_by_x
        self.ct2ct_code_list["code_z_t_by_x"] = code_z_t_by_x
        self.ct2ct_image_list["x_r_c_by_z"] = x_r_c_by_z
        self.ct2ct_image_list["w_t_by_x"] = w_t_by_x
        self.ct2ct_code_list["code_w_t_by_x"] = code_w_t_by_x
        self.ct2ct_image_list["x_r_c_by_w"] = x_r_c_by_w

        self.ct2ct_image_list["l_x_r_c_by_y"] = l_x_r_c_by_y
        self.ct2ct_image_list["l_x_r_c_by_z"] = l_x_r_c_by_z
        self.ct2ct_image_list["l_x_r_c_by_w"] = l_x_r_c_by_w

        self.ct2ct_judge_list["j_x"], self.ct2ct_judge_list["j_x_c"] = j_x, j_x_c
        self.ct2ct_judge_list["j_y_t_by_x"], self.ct2ct_judge_list["j_y_t_c_by_x"] = j_y_t_by_x, j_y_t_c_by_x
        self.ct2ct_judge_list["j_z_t_by_x"], self.ct2ct_judge_list["j_z_t_c_by_x"] = j_z_t_by_x, j_z_t_c_by_x
        self.ct2ct_judge_list["j_w_t_by_x"], self.ct2ct_judge_list["j_w_t_c_by_x"] = j_w_t_by_x, j_w_t_c_by_x

        self.ct2ct_judge_list["j_x_ct_or_mri"] = j_x_ct_or_mri
        self.ct2ct_judge_list["j_y_t_ct_or_mri"] = j_y_t_ct_or_mri
        self.ct2ct_judge_list["j_z_t_ct_or_mri"] = j_z_t_ct_or_mri
        self.ct2ct_judge_list["j_w_t_ct_or_mri"] = j_w_t_ct_or_mri
        self.ct2ct_judge_list["j_code_x_ct_or_mri"] = j_code_x_ct_or_mri

        loss_list = [G_loss, D_loss]

        return loss_list

    def MRI_2_MRI_model(self, l_x, x, c_list):
        c_ct = 0.0
        c_mri = 1.0
        cx = c_list[0]
        cy = c_list[1]
        cz = c_list[2]
        cw = c_list[3]
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)
        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])
        l_x = l_x * 0.25

        mask_x = self.get_mask(x)

        code_x = self.EC_MRI(x)

        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        x_r = self.DC_MRI(tf.concat([code_x, cx_code], axis=-1))

        y_t_by_x = self.DC_MRI(tf.concat([code_x, cy_code], axis=-1))
        code_y_t_by_x = self.EC_MRI(y_t_by_x)
        x_r_c_by_y = self.DC_MRI(tf.concat([code_y_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_y = self.DC_L(code_y_t_by_x)
        l_x_r_c_by_y = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        z_t_by_x = self.DC_MRI(tf.concat([code_x, cz_code], axis=-1))
        code_z_t_by_x = self.EC_MRI(z_t_by_x)
        x_r_c_by_z = self.DC_MRI(tf.concat([code_z_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_z = self.DC_L(code_z_t_by_x)
        l_x_r_c_by_z = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)
        w_t_by_x = self.DC_MRI(tf.concat([code_x, cw_code], axis=-1))
        code_w_t_by_x = self.EC_MRI(w_t_by_x)
        x_r_c_by_w = self.DC_MRI(tf.concat([code_w_t_by_x, cx_code], axis=-1))
        l_prob_x_r_c_by_w = self.DC_L(code_w_t_by_x)
        l_x_r_c_by_w = tf.reshape(
            tf.cast(tf.argmax(l_prob_x_r_c_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        j_x, j_x_c, j_x_ct_or_mri = self.D_M(x)
        j_y_t_by_x, j_y_t_c_by_x, j_y_t_ct_or_mri = self.D_M(y_t_by_x)
        j_z_t_by_x, j_z_t_c_by_x, j_z_t_ct_or_mri = self.D_M(z_t_by_x)
        j_w_t_by_x, j_w_t_c_by_x, j_w_t_ct_or_mri = self.D_M(w_t_by_x)
        j_code_x_ct_or_mri = self.FD_M(code_x)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_x, 1.0) * 45

        D_loss += self.mse_loss(j_y_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_y_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_z_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_z_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_w_t_by_x, 0.0) * 10
        G_loss += self.mse_loss(j_w_t_by_x, 1.0) * 10

        D_loss += self.mse_loss(j_x_c, cx) * 50
        G_loss += self.mse_loss(j_y_t_c_by_x, cy) * 50
        G_loss += self.mse_loss(j_z_t_c_by_x, cz) * 50
        G_loss += self.mse_loss(j_w_t_c_by_x, cw) * 50

        D_loss += self.mse_loss(j_x_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_y_t_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_z_t_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_w_t_ct_or_mri, c_mri) * 50

        D_loss += self.mse_loss(j_code_x_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_code_x_ct_or_mri, c_ct) * 50

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
        G_loss += self.mse_loss(l_x, l_x_r_c_by_y) * 5
        G_loss += self.mse_loss(l_x, l_x_r_c_by_z) * 5
        G_loss += self.mse_loss(l_x, l_x_r_c_by_w) * 5
        G_loss += self.mse_loss(l_x_r_c_by_y, l_x_r_c_by_z)
        G_loss += self.mse_loss(l_x_r_c_by_y, l_x_r_c_by_w)
        G_loss += self.mse_loss(l_x_r_c_by_z, l_x_r_c_by_w)

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_r_c_by_y) * 10
        G_loss += self.mse_loss(x, x_r_c_by_z) * 10
        G_loss += self.mse_loss(x, x_r_c_by_w) * 10
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_z) * 2
        G_loss += self.mse_loss(x_r_c_by_y, x_r_c_by_w) * 2
        G_loss += self.mse_loss(x_r_c_by_z, x_r_c_by_w) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, y_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, z_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, w_t_by_x * mask_x) * 2
        G_loss += self.mse_loss(0.0, x_r * mask_x) * 0.5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_z_t_by_x) * 5
        G_loss += self.mse_loss(code_x, code_w_t_by_x) * 5
        G_loss += self.mse_loss(code_y_t_by_x, code_z_t_by_x)
        G_loss += self.mse_loss(code_y_t_by_x, code_w_t_by_x)
        G_loss += self.mse_loss(code_z_t_by_x, code_w_t_by_x)

        self.mri2mri_image_list["l_x"] = l_x
        self.mri2mri_image_list["x"] = x

        self.mri2mri_prob_list["label_expand_x"] = label_expand_x

        self.mri2mri_image_list["mask_x"] = mask_x

        self.mri2mri_code_list["code_x"] = code_x

        self.mri2mri_prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.mri2mri_image_list["l_f_by_x"] = l_f_by_x

        self.mri2mri_image_list["x_r"] = x_r

        self.mri2mri_image_list["y_t_by_x"] = y_t_by_x
        self.mri2mri_code_list["code_y_t_by_x"] = code_y_t_by_x
        self.mri2mri_image_list["x_r_c_by_y"] = x_r_c_by_y
        self.mri2mri_image_list["z_t_by_x"] = z_t_by_x
        self.mri2mri_code_list["code_z_t_by_x"] = code_z_t_by_x
        self.mri2mri_image_list["x_r_c_by_z"] = x_r_c_by_z
        self.mri2mri_image_list["w_t_by_x"] = w_t_by_x
        self.mri2mri_code_list["code_w_t_by_x"] = code_w_t_by_x
        self.mri2mri_image_list["x_r_c_by_w"] = x_r_c_by_w

        self.mri2mri_image_list["l_x_r_c_by_y"] = l_x_r_c_by_y
        self.mri2mri_image_list["l_x_r_c_by_z"] = l_x_r_c_by_z
        self.mri2mri_image_list["l_x_r_c_by_w"] = l_x_r_c_by_w

        self.mri2mri_judge_list["j_x"], self.mri2mri_judge_list["j_x_c"] = j_x, j_x_c
        self.mri2mri_judge_list["j_y_t_by_x"], self.mri2mri_judge_list["j_y_t_c_by_x"] = j_y_t_by_x, j_y_t_c_by_x
        self.mri2mri_judge_list["j_z_t_by_x"], self.mri2mri_judge_list["j_z_t_c_by_x"] = j_z_t_by_x, j_z_t_c_by_x
        self.mri2mri_judge_list["j_w_t_by_x"], self.mri2mri_judge_list["j_w_t_c_by_x"] = j_w_t_by_x, j_w_t_c_by_x

        self.mri2mri_judge_list["j_x_ct_or_mri"] = j_x_ct_or_mri
        self.mri2mri_judge_list["j_y_t_ct_or_mri"] = j_y_t_ct_or_mri
        self.mri2mri_judge_list["j_z_t_ct_or_mri"] = j_z_t_ct_or_mri
        self.mri2mri_judge_list["j_w_t_ct_or_mri"] = j_w_t_ct_or_mri
        self.mri2mri_judge_list["j_code_x_ct_or_mri"] = j_code_x_ct_or_mri

        loss_list = [G_loss, D_loss]

        return loss_list

    def CT_2_MRI_model(self, l_m, m, cm):
        c_ct = 0.0
        c_mri = 1.0
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cm_code = self.ones_code * tf.one_hot(tf.cast(cm, dtype=tf.int32), depth=4)
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)
        label_expand_m = tf.reshape(tf.one_hot(tf.cast(l_m, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])

        l_m = l_m * 0.25

        mask_m = self.get_mask(m)

        code_m = self.EC_CT(m)

        l_f_prob_by_m = self.DC_L(code_m)
        l_f_by_m = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_m, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        x_t_by_m = self.DC_MRI(tf.concat([code_m, cx_code], axis=-1))
        code_x_t_by_m = self.EC_MRI(x_t_by_m)
        m_r_c_by_x = self.DC_CT(tf.concat([code_x_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_x = self.DC_L(code_x_t_by_m)
        l_m_r_c_by_x = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        y_t_by_m = self.DC_MRI(tf.concat([code_m, cy_code], axis=-1))
        code_y_t_by_m = self.EC_MRI(y_t_by_m)
        m_r_c_by_y = self.DC_CT(tf.concat([code_y_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_y = self.DC_L(code_y_t_by_m)
        l_m_r_c_by_y = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        z_t_by_m = self.DC_MRI(tf.concat([code_m, cz_code], axis=-1))
        code_z_t_by_m = self.EC_MRI(z_t_by_m)
        m_r_c_by_z = self.DC_CT(tf.concat([code_z_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_z = self.DC_L(code_z_t_by_m)
        l_m_r_c_by_z = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        w_t_by_m = self.DC_MRI(tf.concat([code_m, cw_code], axis=-1))
        code_w_t_by_m = self.EC_MRI(w_t_by_m)
        m_r_c_by_w = self.DC_CT(tf.concat([code_w_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_w = self.DC_L(code_w_t_by_m)
        l_m_r_c_by_w = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        j_m, j_m_c, j_m_ct_or_mri = self.D_M(m)
        j_x_t_by_m, j_x_t_c_by_m, j_x_t_ct_or_mri = self.D_M(x_t_by_m)
        j_y_t_by_m, j_y_t_c_by_m, j_y_t_ct_or_mri = self.D_M(y_t_by_m)
        j_z_t_by_m, j_z_t_c_by_m, j_z_t_ct_or_mri = self.D_M(z_t_by_m)
        j_w_t_by_m, j_w_t_c_by_m, j_w_t_ct_or_mri = self.D_M(w_t_by_m)
        j_code_x_ct_or_mri = self.FD_M(code_m)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_m, 1.0) * 45

        D_loss += self.mse_loss(j_x_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_x_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_y_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_y_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_z_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_z_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_w_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_w_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_m_c, cm) * 50
        G_loss += self.mse_loss(j_x_t_c_by_m, cx) * 50
        G_loss += self.mse_loss(j_y_t_c_by_m, cy) * 50
        G_loss += self.mse_loss(j_z_t_c_by_m, cz) * 50
        G_loss += self.mse_loss(j_w_t_c_by_m, cw) * 50

        D_loss += self.mse_loss(j_m_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_x_t_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_y_t_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_z_t_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_w_t_ct_or_mri, c_mri) * 50

        D_loss += self.mse_loss(j_code_x_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_code_x_ct_or_mri, c_mri) * 50

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_m[:, :, :, 0],
                                l_f_prob_by_m[:, :, :, 0]) \
                  + self.mse_loss(label_expand_m[:, :, :, 1],
                                  l_f_prob_by_m[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_m[:, :, :, 2],
                                  l_f_prob_by_m[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_m[:, :, :, 3],
                                  l_f_prob_by_m[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_m[:, :, :, 4],
                                  l_f_prob_by_m[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_m, l_f_by_m) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_x) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_y) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_z) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_w) * 5
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_y)
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_z)
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_w)
        G_loss += self.mse_loss(l_m_r_c_by_y, l_m_r_c_by_z)
        G_loss += self.mse_loss(l_m_r_c_by_y, l_m_r_c_by_w)
        G_loss += self.mse_loss(l_m_r_c_by_z, l_m_r_c_by_w)

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(m, m_r_c_by_x) * 10
        G_loss += self.mse_loss(m, m_r_c_by_y) * 10
        G_loss += self.mse_loss(m, m_r_c_by_z) * 10
        G_loss += self.mse_loss(m, m_r_c_by_w) * 10
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_y) * 2
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_z) * 2
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_w) * 2
        G_loss += self.mse_loss(m_r_c_by_y, m_r_c_by_z) * 2
        G_loss += self.mse_loss(m_r_c_by_y, m_r_c_by_w) * 2
        G_loss += self.mse_loss(m_r_c_by_z, m_r_c_by_w) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, y_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, z_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, w_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, m_r_c_by_x * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_y * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_z * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_w * mask_m)

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_m, code_x_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_y_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_z_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_w_t_by_m) * 5
        G_loss += self.mse_loss(code_x_t_by_m, code_y_t_by_m)
        G_loss += self.mse_loss(code_x_t_by_m, code_z_t_by_m)
        G_loss += self.mse_loss(code_x_t_by_m, code_w_t_by_m)
        G_loss += self.mse_loss(code_y_t_by_m, code_z_t_by_m)
        G_loss += self.mse_loss(code_y_t_by_m, code_w_t_by_m)
        G_loss += self.mse_loss(code_z_t_by_m, code_w_t_by_m)

        self.ct2mri_image_list["l_m"] = l_m
        self.ct2mri_image_list["m"] = m

        self.ct2mri_prob_list["label_expand_m"] = label_expand_m

        self.ct2mri_image_list["mask_m"] = mask_m

        self.ct2mri_code_list["code_m"] = code_m

        self.ct2mri_prob_list["l_f_prob_by_m"] = l_f_prob_by_m
        self.ct2mri_image_list["l_f_by_m"] = l_f_by_m

        self.ct2mri_image_list["x_t_by_m"] = x_t_by_m
        self.ct2mri_code_list["code_x_t_by_m"] = code_x_t_by_m
        self.ct2mri_image_list["m_r_c_by_x"] = m_r_c_by_x

        self.ct2mri_image_list["y_t_by_m"] = y_t_by_m
        self.ct2mri_code_list["code_y_t_by_m"] = code_y_t_by_m
        self.ct2mri_image_list["m_r_c_by_y"] = m_r_c_by_y

        self.ct2mri_image_list["z_t_by_m"] = z_t_by_m
        self.ct2mri_code_list["code_z_t_by_m"] = code_z_t_by_m
        self.ct2mri_image_list["m_r_c_by_z"] = m_r_c_by_z

        self.ct2mri_image_list["w_t_by_m"] = w_t_by_m
        self.ct2mri_code_list["code_w_t_by_m"] = code_w_t_by_m
        self.ct2mri_image_list["m_r_c_by_w"] = m_r_c_by_w

        self.ct2mri_image_list["l_m_r_c_by_x"] = l_m_r_c_by_x
        self.ct2mri_image_list["l_m_r_c_by_y"] = l_m_r_c_by_y
        self.ct2mri_image_list["l_m_r_c_by_z"] = l_m_r_c_by_z
        self.ct2mri_image_list["l_m_r_c_by_w"] = l_m_r_c_by_w

        self.ct2mri_judge_list["j_m"], self.ct2mri_judge_list["j_m_c"] = j_m, j_m_c
        self.ct2mri_judge_list["j_x_t_by_m"], self.ct2mri_judge_list["j_x_t_c_by_m"] = j_x_t_by_m, j_x_t_c_by_m
        self.ct2mri_judge_list["j_y_t_by_m"], self.ct2mri_judge_list["j_y_t_c_by_m"] = j_y_t_by_m, j_y_t_c_by_m
        self.ct2mri_judge_list["j_z_t_by_m"], self.ct2mri_judge_list["j_z_t_c_by_m"] = j_z_t_by_m, j_z_t_c_by_m
        self.ct2mri_judge_list["j_w_t_by_m"], self.ct2mri_judge_list["j_w_t_c_by_m"] = j_w_t_by_m, j_w_t_c_by_m

        self.ct2mri_judge_list["j_m_ct_or_mri"] = j_m_ct_or_mri
        self.ct2mri_judge_list["j_x_t_ct_or_mri"] = j_x_t_ct_or_mri
        self.ct2mri_judge_list["j_y_t_ct_or_mri"] = j_y_t_ct_or_mri
        self.ct2mri_judge_list["j_z_t_ct_or_mri"] = j_z_t_ct_or_mri
        self.ct2mri_judge_list["j_w_t_ct_or_mri"] = j_w_t_ct_or_mri
        self.ct2mri_judge_list["j_code_x_ct_or_mri"] = j_code_x_ct_or_mri

        loss_list = [G_loss, D_loss]

        return loss_list

    def MRI_2_CT_model(self, l_m, m, cm):
        c_ct = 0.0
        c_mri = 1.0
        cx = 0.0
        cy = 1.0
        cz = 2.0
        cw = 3.0
        cm_code = self.ones_code * tf.one_hot(tf.cast(cm, dtype=tf.int32), depth=4)
        cx_code = self.ones_code * tf.one_hot(tf.cast(cx, dtype=tf.int32), depth=4)
        cy_code = self.ones_code * tf.one_hot(tf.cast(cy, dtype=tf.int32), depth=4)
        cz_code = self.ones_code * tf.one_hot(tf.cast(cz, dtype=tf.int32), depth=4)
        cw_code = self.ones_code * tf.one_hot(tf.cast(cw, dtype=tf.int32), depth=4)
        label_expand_m = tf.reshape(tf.one_hot(tf.cast(l_m, dtype=tf.int32), axis=-1, depth=5),
                                    shape=[self.input_shape[0], self.input_shape[1],
                                           self.input_shape[2], 5])

        l_m = l_m * 0.25

        mask_m = self.get_mask(m)

        code_m = self.EC_MRI(m)

        l_f_prob_by_m = self.DC_L(code_m)
        l_f_by_m = tf.reshape(
            tf.cast(tf.argmax(l_f_prob_by_m, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        x_t_by_m = self.DC_CT(tf.concat([code_m, cx_code], axis=-1))
        code_x_t_by_m = self.EC_CT(x_t_by_m)
        m_r_c_by_x = self.DC_MRI(tf.concat([code_x_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_x = self.DC_L(code_x_t_by_m)
        l_m_r_c_by_x = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_x, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        y_t_by_m = self.DC_CT(tf.concat([code_m, cy_code], axis=-1))
        code_y_t_by_m = self.EC_CT(y_t_by_m)
        m_r_c_by_y = self.DC_MRI(tf.concat([code_y_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_y = self.DC_L(code_y_t_by_m)
        l_m_r_c_by_y = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_y, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        z_t_by_m = self.DC_CT(tf.concat([code_m, cz_code], axis=-1))
        code_z_t_by_m = self.EC_CT(z_t_by_m)
        m_r_c_by_z = self.DC_MRI(tf.concat([code_z_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_z = self.DC_L(code_z_t_by_m)
        l_m_r_c_by_z = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_z, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        w_t_by_m = self.DC_CT(tf.concat([code_m, cw_code], axis=-1))
        code_w_t_by_m = self.EC_CT(w_t_by_m)
        m_r_c_by_w = self.DC_MRI(tf.concat([code_w_t_by_m, cm_code], axis=-1))
        l_prob_y_r_c_by_w = self.DC_L(code_w_t_by_m)
        l_m_r_c_by_w = tf.reshape(
            tf.cast(tf.argmax(l_prob_y_r_c_by_w, axis=-1), dtype=tf.float32) * 0.25,
            shape=self.input_shape)

        j_m, j_m_c, j_m_ct_or_mri = self.D_M(m)
        j_x_t_by_m, j_x_t_c_by_m, j_x_t_ct_or_mri = self.D_M(x_t_by_m)
        j_y_t_by_m, j_y_t_c_by_m, j_y_t_ct_or_mri = self.D_M(y_t_by_m)
        j_z_t_by_m, j_z_t_c_by_m, j_z_t_ct_or_mri = self.D_M(z_t_by_m)
        j_w_t_by_m, j_w_t_c_by_m, j_w_t_ct_or_mri = self.D_M(w_t_by_m)
        j_code_x_ct_or_mri = self.FD_M(code_m)

        D_loss = 0.0
        G_loss = 0.0
        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_loss += self.mse_loss(j_m, 1.0) * 45

        D_loss += self.mse_loss(j_x_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_x_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_y_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_y_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_z_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_z_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_w_t_by_m, 0.0) * 10
        G_loss += self.mse_loss(j_w_t_by_m, 1.0) * 10

        D_loss += self.mse_loss(j_m_c, cm) * 50
        G_loss += self.mse_loss(j_x_t_c_by_m, cx) * 50
        G_loss += self.mse_loss(j_y_t_c_by_m, cy) * 50
        G_loss += self.mse_loss(j_z_t_c_by_m, cz) * 50
        G_loss += self.mse_loss(j_w_t_c_by_m, cw) * 50

        D_loss += self.mse_loss(j_m_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_x_t_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_y_t_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_z_t_ct_or_mri, c_ct) * 50
        G_loss += self.mse_loss(j_w_t_ct_or_mri, c_ct) * 50

        D_loss += self.mse_loss(j_code_x_ct_or_mri, c_mri) * 50
        G_loss += self.mse_loss(j_code_x_ct_or_mri, c_ct) * 50

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_m[:, :, :, 0],
                                l_f_prob_by_m[:, :, :, 0]) \
                  + self.mse_loss(label_expand_m[:, :, :, 1],
                                  l_f_prob_by_m[:, :, :, 1]) * 5 \
                  + self.mse_loss(label_expand_m[:, :, :, 2],
                                  l_f_prob_by_m[:, :, :, 2]) * 15 \
                  + self.mse_loss(label_expand_m[:, :, :, 3],
                                  l_f_prob_by_m[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_m[:, :, :, 4],
                                  l_f_prob_by_m[:, :, :, 4]) * 15
        G_loss += self.mse_loss(l_m, l_f_by_m) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_x) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_y) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_z) * 5
        G_loss += self.mse_loss(l_m, l_m_r_c_by_w) * 5
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_y)
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_z)
        G_loss += self.mse_loss(l_m_r_c_by_x, l_m_r_c_by_w)
        G_loss += self.mse_loss(l_m_r_c_by_y, l_m_r_c_by_z)
        G_loss += self.mse_loss(l_m_r_c_by_y, l_m_r_c_by_w)
        G_loss += self.mse_loss(l_m_r_c_by_z, l_m_r_c_by_w)

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(m, m_r_c_by_x) * 10
        G_loss += self.mse_loss(m, m_r_c_by_y) * 10
        G_loss += self.mse_loss(m, m_r_c_by_z) * 10
        G_loss += self.mse_loss(m, m_r_c_by_w) * 10
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_y) * 2
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_z) * 2
        G_loss += self.mse_loss(m_r_c_by_x, m_r_c_by_w) * 2
        G_loss += self.mse_loss(m_r_c_by_y, m_r_c_by_z) * 2
        G_loss += self.mse_loss(m_r_c_by_y, m_r_c_by_w) * 2
        G_loss += self.mse_loss(m_r_c_by_z, m_r_c_by_w) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, y_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, z_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, w_t_by_m * mask_m) * 2
        G_loss += self.mse_loss(0.0, m_r_c_by_x * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_y * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_z * mask_m)
        G_loss += self.mse_loss(0.0, m_r_c_by_w * mask_m)

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_m, code_x_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_y_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_z_t_by_m) * 5
        G_loss += self.mse_loss(code_m, code_w_t_by_m) * 5
        G_loss += self.mse_loss(code_x_t_by_m, code_y_t_by_m)
        G_loss += self.mse_loss(code_x_t_by_m, code_z_t_by_m)
        G_loss += self.mse_loss(code_x_t_by_m, code_w_t_by_m)
        G_loss += self.mse_loss(code_y_t_by_m, code_z_t_by_m)
        G_loss += self.mse_loss(code_y_t_by_m, code_w_t_by_m)
        G_loss += self.mse_loss(code_z_t_by_m, code_w_t_by_m)

        self.mri2ct_image_list["l_m"] = l_m
        self.mri2ct_image_list["m"] = m

        self.mri2ct_prob_list["label_expand_m"] = label_expand_m

        self.mri2ct_image_list["mask_m"] = mask_m

        self.mri2ct_code_list["code_m"] = code_m

        self.mri2ct_prob_list["l_f_prob_by_m"] = l_f_prob_by_m
        self.mri2ct_image_list["l_f_by_m"] = l_f_by_m

        self.mri2ct_image_list["x_t_by_m"] = x_t_by_m
        self.mri2ct_code_list["code_x_t_by_m"] = code_x_t_by_m
        self.mri2ct_image_list["m_r_c_by_x"] = m_r_c_by_x

        self.mri2ct_image_list["y_t_by_m"] = y_t_by_m
        self.mri2ct_code_list["code_y_t_by_m"] = code_y_t_by_m
        self.mri2ct_image_list["m_r_c_by_y"] = m_r_c_by_y

        self.mri2ct_image_list["z_t_by_m"] = z_t_by_m
        self.mri2ct_code_list["code_z_t_by_m"] = code_z_t_by_m
        self.mri2ct_image_list["m_r_c_by_z"] = m_r_c_by_z

        self.mri2ct_image_list["w_t_by_m"] = w_t_by_m
        self.mri2ct_code_list["code_w_t_by_m"] = code_w_t_by_m
        self.mri2ct_image_list["m_r_c_by_w"] = m_r_c_by_w

        self.mri2ct_image_list["l_m_r_c_by_x"] = l_m_r_c_by_x
        self.mri2ct_image_list["l_m_r_c_by_y"] = l_m_r_c_by_y
        self.mri2ct_image_list["l_m_r_c_by_z"] = l_m_r_c_by_z
        self.mri2ct_image_list["l_m_r_c_by_w"] = l_m_r_c_by_w

        self.mri2ct_judge_list["j_m"], self.mri2ct_judge_list["j_m_c"] = j_m, j_m_c
        self.mri2ct_judge_list["j_x_t_by_m"], self.mri2ct_judge_list["j_x_t_c_by_m"] = j_x_t_by_m, j_x_t_c_by_m
        self.mri2ct_judge_list["j_y_t_by_m"], self.mri2ct_judge_list["j_y_t_c_by_m"] = j_y_t_by_m, j_y_t_c_by_m
        self.mri2ct_judge_list["j_z_t_by_m"], self.mri2ct_judge_list["j_z_t_c_by_m"] = j_z_t_by_m, j_z_t_c_by_m
        self.mri2ct_judge_list["j_w_t_by_m"], self.mri2ct_judge_list["j_w_t_c_by_m"] = j_w_t_by_m, j_w_t_c_by_m

        self.mri2ct_judge_list["j_m_ct_or_mri"] = j_m_ct_or_mri
        self.mri2ct_judge_list["j_x_t_ct_or_mri"] = j_x_t_ct_or_mri
        self.mri2ct_judge_list["j_y_t_ct_or_mri"] = j_y_t_ct_or_mri
        self.mri2ct_judge_list["j_z_t_ct_or_mri"] = j_z_t_ct_or_mri
        self.mri2ct_judge_list["j_w_t_ct_or_mri"] = j_w_t_ct_or_mri
        self.mri2ct_judge_list["j_code_x_ct_or_mri"] = j_code_x_ct_or_mri

        loss_list = [G_loss, D_loss]

        return loss_list

    def model(self, l_ct, ct, c_ct_list,
              l_mri, mri, c_mri_list,
              l_m_ct, m_ct, cm_ct,
              l_m_mri, m_mri, cm_mri):
        loss_list_1 = self.CT_2_CT_model(l_ct, ct, c_ct_list)
        loss_list_2 = self.MRI_2_MRI_model(l_mri, mri, c_mri_list)
        loss_list_3 = self.CT_2_MRI_model(l_m_ct, m_ct, cm_ct)
        loss_list_4 = self.MRI_2_CT_model(l_m_mri, m_mri, cm_mri)
        G_loss = loss_list_1[0] + loss_list_2[0] + loss_list_3[0] + loss_list_4[0]
        D_loss = loss_list_1[1] + loss_list_2[1] + loss_list_3[1] + loss_list_4[1]
        loss_list = [G_loss, D_loss]

        return loss_list

    def get_variables(self):
        return [self.DC_L.variables
                + self.EC_CT.variables
                + self.DC_CT.variables
                + self.EC_MRI.variables
                + self.DC_MRI.variables
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
            tf.summary.image('judge/' + key, judge_dirct[key])

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_dirct):
        self.name_code_list_true = ["code_x", "code_x", "code_x",
                                    "code_y", "code_y", "code_y",
                                    "code_z", "code_z", "code_z",
                                    "code_w", "code_w", "code_w"]
        self.name_code_list_false = ["code_y_t_by_x", "code_z_t_by_x", "code_w_t_by_x",
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
        self.name_list_true = ["x", "x", "x", "x",
                               "y", "z", "w"
                               ]
        self.name_list_false = ["x_r", "x_t_by_y", "x_t_by_z", "x_t_by_w",
                                "y_t_by_x", "z_t_by_x", "w_t_by_x"
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
