# _*_ coding:utf-8 _*_
import tensorflow as tf
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

        self.EC_L = Encoder('EC_L', ngf=ngf)
        self.DC_L_X = Decoder('DC_L_X', ngf=ngf, output_channl=5)
        self.DC_L_Y = Decoder('DC_L_Y', ngf=ngf, output_channl=5)
        self.DC_L_Z = Decoder('DC_L_Z', ngf=ngf, output_channl=5)
        self.DC_L_W = Decoder('DC_L_W', ngf=ngf, output_channl=5)

    def model(self, l_x, l_y, l_z, l_w, x, y, z, w):
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
        l_x = l_x * 0.25
        l_y = l_y * 0.25
        l_z = l_z * 0.25
        l_w = l_w * 0.25

        code_x = self.EC_L(x)
        code_y = self.EC_L(y)
        code_z = self.EC_L(z)
        code_w = self.EC_L(w)

        l_f_prob_by_x = self.DC_L_X(code_x)
        l_f_prob_by_y = self.DC_L_Y(code_y)
        l_f_prob_by_z = self.DC_L_Z(code_z)
        l_f_prob_by_w = self.DC_L_W(code_w)
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

        G_loss = 0.0
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

        self.image_list["l_x"] = l_x
        self.image_list["l_y"] = l_y
        self.image_list["l_z"] = l_z
        self.image_list["l_w"] = l_w
        self.image_list["x"] = x
        self.image_list["y"] = y
        self.image_list["z"] = z
        self.image_list["w"] = w

        self.prob_list["label_expand_x"] = label_expand_x
        self.prob_list["label_expand_y"] = label_expand_y
        self.prob_list["label_expand_z"] = label_expand_z
        self.prob_list["label_expand_w"] = label_expand_w

        self.prob_list["l_f_prob_by_x"] = l_f_prob_by_x
        self.prob_list["l_f_prob_by_y"] = l_f_prob_by_y
        self.prob_list["l_f_prob_by_z"] = l_f_prob_by_z
        self.prob_list["l_f_prob_by_w"] = l_f_prob_by_w

        self.image_list["l_f_by_x"] = l_f_by_x
        self.image_list["l_f_by_y"] = l_f_by_y
        self.image_list["l_f_by_z"] = l_f_by_z
        self.image_list["l_f_by_w"] = l_f_by_w

        return G_loss

    def get_variables(self):
        return [self.EC_L.variables
                + self.DC_L_X.variables
                + self.DC_L_Y.variables
                + self.DC_L_Z.variables
                + self.DC_L_W.variables
                ]

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

    def loss_summary(self, G_loss):
        tf.summary.scalar('loss/G_loss', G_loss)

    def evaluation(self, image_dirct):
        self.name_list_true = ["l_x", "l_y", "l_z", "l_w"]
        self.name_list_false = ["l_f_by_x", "l_f_by_y", "l_f_by_z", "l_f_by_w"]
        dice_score_list = []
        mse_list = []
        for i in range(len(self.name_list_true)):
            dice_score_list.append(
                self.dice_score(image_dirct[self.name_list_true[i]] * 4, image_dirct[self.name_list_false[i]] * 4))
            mse_list.append(
                self.mse_loss(image_dirct[self.name_list_true[i]] * 4, image_dirct[self.name_list_false[i]] * 4))
        return dice_score_list, mse_list

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

    def dice_score(self, output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
        inse = tf.reduce_sum(output * target, axis=axis)
        if loss_type == 'jaccard':
            l = tf.reduce_sum(output * output, axis=axis)
            r = tf.reduce_sum(target * target, axis=axis)
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(output, axis=axis)
            r = tf.reduce_sum(target, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)
        return dice

    def cos_score(self, output, target, axis=(1, 2, 3), smooth=1e-5):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(output), axis))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(target), axis))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(output, target), axis)
        score = tf.reduce_mean(tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + smooth))
        return score

    def euclidean_distance(self, output, target, axis=(1, 2, 3)):
        euclidean = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(output - target), axis)))
        return euclidean

    def MSE(self, output, target):
        mse = tf.reduce_mean(tf.square(output - target))
        return mse

    def MAE(self, output, target):
        mae = tf.reduce_mean(tf.abs(output - target))
        return mae

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
