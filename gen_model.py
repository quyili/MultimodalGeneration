# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
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
        self.input_shape = [batch_size, image_size[0], image_size[1], image_size[2]]
        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.DC_X = Decoder('DC_X', ngf=ngf)
        self.DC_Y = Decoder('DC_Y', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)
        self.D_X = Discriminator('D_X', ngf=ngf)
        self.D_Y = Discriminator('D_Y', ngf=ngf)
        self.x = tf.placeholder(tf.float32, shape=self.input_shape)
        self.y = tf.placeholder(tf.float32, shape=self.input_shape)
        self.rm = tf.placeholder(tf.float32, shape=self.input_shape)
        self.mask = tf.placeholder(tf.float32, shape=self.input_shape)
        self.label_expand = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 6])

    def model(self):
        l_input = tf.reshape(tf.cast(tf.argmax(self.label_expand, axis=-1), dtype=tf.float32) * 0.2,
                             shape=self.input_shape)
        rm_input = tf.concat([
            tf.reshape(self.rm[:, :, :, 0] * self.label_expand[:, :, :, 1], shape=self.input_shape),
            tf.reshape(self.rm[:, :, :, 0] * self.label_expand[:, :, :, 2], shape=self.input_shape),
            tf.reshape(self.rm[:, :, :, 0] * self.label_expand[:, :, :, 3], shape=self.input_shape),
            tf.reshape(self.rm[:, :, :, 0] * self.label_expand[:, :, :, 4], shape=self.input_shape),
            tf.reshape(self.rm[:, :, :, 0] * self.label_expand[:, :, :, 5], shape=self.input_shape)], axis=-1)

        # R -> X_G,Y_G,L
        code_rm = self.EC_R(rm_input)
        x_g = self.DC_X(code_rm)
        y_g = self.DC_Y(code_rm)
        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)
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

        # X -> X_R
        code_x = self.EC_X(self.x)
        x_r = self.DC_X(code_x)
        # Y -> Y_R
        code_y = self.EC_Y(self.y)
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

        j_x = self.D_X(self.x)
        j_x_g = self.D_X(x_g)
        j_y = self.D_Y(self.y)
        j_y_g = self.D_Y(y_g)

        D_loss = self.mse_loss(j_x, 1.0) * 5
        D_loss += self.mse_loss(j_x_g, 0.0) * 5
        G_loss = self.mse_loss(j_x_g, 1.0) * 5
        G_loss += self.mse_loss(j_x_g, j_x) * 10
        G_loss += self.mse_loss(0.0, x_g * self.label_expand[0]) * 0.5

        D_loss += self.mse_loss(j_y, 1.0) * 5
        D_loss += self.mse_loss(j_y_g, 0.0) * 5
        G_loss += self.mse_loss(j_y_g, 1.0) * 5
        G_loss += self.mse_loss(j_y_g, j_y) * 10
        G_loss += self.mse_loss(0.0, y_g * self.label_expand[0]) * 0.5

        G_loss += self.mse_loss(self.label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) * 5 \
                  + self.mse_loss(self.label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 5], l_g_prob[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l_input, l_g)

        G_loss += self.mse_loss(self.label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(self.label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 5], l_g_prob_by_x[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l_input, l_g_by_x)

        G_loss += self.mse_loss(self.label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(self.label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 5], l_g_prob_by_y[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l_input, l_g_by_y)

        G_loss += self.mse_loss(self.label_expand[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(self.label_expand[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l_input, l_f_by_x)*5

        G_loss += self.mse_loss(self.label_expand[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) * 0.5 \
                  + self.mse_loss(self.label_expand[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(self.label_expand[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 80 \
                  + self.mse_loss(self.label_expand[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 80
        G_loss += self.mse_loss(l_input, l_f_by_y)*5

        G_loss += (self.mse_loss(code_x[0], code_y[0]) * 0.5 + self.mse_loss(code_x[1], code_y[1])) * 0.5
        G_loss += (self.mse_loss(code_rm[0], code_x_g[0]) * 0.5 + self.mse_loss(code_rm[1], code_x_g[1])) * 0.8
        G_loss += (self.mse_loss(code_rm[0], code_y_g[0]) * 0.5 + self.mse_loss(code_rm[1], code_y_g[1])) * 0.8

        G_loss += self.mse_loss(l_g_by_x, l_g_by_y) * 0.7
        G_loss += (self.mse_loss(code_x_g[0], code_y_g[0]) * 0.5 + self.mse_loss(code_x_g[1], code_y_g[1])) * 0.5

        G_loss += self.mse_loss(y_g, y_g_t)*2
        G_loss += self.mse_loss(x_g, x_g_t)*2
        G_loss += self.mse_loss(self.x, x_r)
        G_loss += self.mse_loss(self.y, y_r)
        G_loss += self.mse_loss(self.x, x_t)*2
        G_loss += self.mse_loss(self.y, y_t)*2

        evluation_list = [D_loss, G_loss]

        return l_input, rm_input, \
               x_g, y_g,x_g_t, y_g_t,x_r, y_r, x_t, y_t,\
               l_g, l_f_by_x, l_f_by_y, l_g_by_x, l_g_by_y, \
               G_loss, D_loss, evluation_list

    def optimize(self, G_loss, D_loss, ):
        def make_optimizer(loss, variables, name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
                    .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        global_step = tf.Variable(0, trainable=False)
        G_optimizer = make_optimizer(G_loss,
                                     self.EC_R.variables
                                     + self.EC_X.variables
                                     + self.EC_Y.variables
                                     + self.DC_X.variables
                                     + self.DC_Y.variables
                                     + self.DC_L.variables
                                     ,
                                     name='Adam_G')
        D_optimizer = make_optimizer(D_loss, self.D_X.variables + self.D_Y.variables, name='Adam_D')

        with tf.control_dependencies(
                [G_optimizer,
                 D_optimizer]):
            return tf.no_op(name='optimizers')

    def mse_loss(self, x, y):
        """ supervised loss (L2 norm)
        """
        loss = tf.reduce_mean(tf.square(x - y))
        return loss
