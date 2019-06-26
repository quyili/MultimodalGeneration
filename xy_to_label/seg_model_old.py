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
        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.DC_X_L = Decoder('DC_X_L', ngf=ngf, output_channl=6)
        self.DC_Y_L = Decoder('DC_Y_L', ngf=ngf, output_channl=6)

    def model(self, x, y, label_expand):
        # L
        l = tf.reshape(tf.cast(tf.argmax(label_expand, axis=-1), dtype=tf.float32) * 0.2,
                       shape=self.input_shape)

        # X -> L_X
        code_x = self.EC_X(x)
        l_f_prob_by_x = self.DC_X_L(code_x)
        l_f_by_x = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y -> L_Y
        code_y = self.EC_Y(y)
        l_f_prob_by_y = self.DC_Y_L(code_y)
        l_f_by_y = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)

        # X模态图分割训练的有监督损失
        G_loss = self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) \
                 + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) \
                 + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 5 \
                 + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 15 \
                 + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 15 \
                 + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l, l_f_by_x) * 10

        # Y模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) \
                  + self.mse_loss(label_expand[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 15 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l, l_f_by_y) * 10

        image_list = [x, y, l, l_f_by_x, l_f_by_y]

        return image_list, G_loss

    def get_variables(self):
        variables = self.EC_X.variables + self.EC_Y.variables + self.DC_X_L.variables + self.DC_Y_L.variables
        return variables

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step

        G_optimizer = make_optimizer(name='Adam_G')

        return G_optimizer

    def loss_summary(self, loss):
        G_loss = loss
        tf.summary.scalar('loss', G_loss)

    def evaluation(self, image_list):
        x, y, l_input, l_f_by_x, l_f_by_y = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4]
        list = [self.iou(l_input, l_f_by_x), self.iou(l_input, l_f_by_y)]
        return list

    def evaluation_summary(self, evluation_list):
        tf.summary.scalar('evaluation/IOU/l_input__VS__l_f_by_x', evluation_list[0])
        tf.summary.scalar('evaluation/IOU/l_input__VS__l_f_by_y', evluation_list[1])

    def image_summary(self, image_list):
        x, y, l_input, l_f_by_x, l_f_by_y = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4]

        tf.summary.image('image/x', x)
        tf.summary.image('image/y', y)
        tf.summary.image('image/l_input', l_input)
        tf.summary.image('image/l_f_by_x', l_f_by_x)
        tf.summary.image('image/l_f_by_y', l_f_by_y)

    def iou(self, label, predict):
        iou_op = tf.metrics.mean_iou(label, predict, 6)
        mean_iou = iou_op[0]
        return mean_iou

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss

    # def ssim_loss(self, x, y):
    #     """ supervised loss (L2 norm)
    #     """
    #     loss = (1.0 - self.SSIM(x, y)) * 20
    #     return loss
    #
    # def PSNR(self, output, target):
    #     psnr = tf.reduce_mean(tf.image.psnr(output, target, max_val=1.0, name="psnr"))
    #     return psnr
    #
    # def SSIM(self, output, target):
    #     ssim = tf.reduce_mean(tf.image.ssim(output, target, max_val=1.0))
    #     return ssim

    def norm(self, input):
        output = (input - tf.reduce_min(input, axis=[1, 2, 3])
                  ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
        return output
