# _*_ coding:utf-8 _*_
import tensorflow as tf
from discriminator import Discriminator
from feature_discriminator import FeatureDiscriminator
from encoder import Encoder
from shared_decoder import SDecoder
from m_decoder import MDecoder
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
        self.image_list = []
        self.code_list = []
        self.judge_list = []

        self.EC_R = Encoder('EC_R', ngf=ngf)
        self.DC_L = Decoder('DC_L', ngf=ngf, output_channl=6)

        self.EC_X = Encoder('EC_X', ngf=ngf)
        self.EC_Y = Encoder('EC_Y', ngf=ngf)
        self.EC_Z = Encoder('EC_Z', ngf=ngf)
        self.EC_W = Encoder('EC_W', ngf=ngf)

        self.SDC = SDecoder('SDC', ngf=ngf)

        self.DC_X = MDecoder('DC_X', ngf=ngf)
        self.DC_Y = MDecoder('DC_Y', ngf=ngf)
        self.DC_Z = MDecoder('DC_Z', ngf=ngf)
        self.DC_W = MDecoder('DC_W', ngf=ngf)

        self.D_M = Discriminator('D_M', ngf=ngf)
        self.FD_R = FeatureDiscriminator('FD_R', ngf=ngf)

    def get_f(self, x):
        f = self.norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))
        f = f - tf.reduce_mean(f, axis=[1, 2, 3])
        f = self.ones* tf.cast(f > 0.1, dtype=tf.float32)
        return f
    def gen(self,f,l,cx,cy,EC_X, EC_Y, DC_X, DC_Y):
        label_expand = tf.reshape(tf.one_hot(tf.cast(l, dtype=tf.int32), axis=-1, depth=6),
                                  shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        f_rm_expand = tf.concat([
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 0], shape=self.input_shape)
            + tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 1], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 2], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 3], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 4], shape=self.input_shape) + f * 0.8,
            tf.reshape(self.ones[:, :, :, 0] * 0.2 * label_expand[:, :, :, 5], shape=self.input_shape) + f * 0.8], axis=-1)

        # F_RM -> X_G,Y_G,L_G
        code_rm = self.EC_R(f_rm_expand)
        x_g = DC_X(self.SDC(code_rm))
        y_g = DC_Y(self.SDC(code_rm))
        l_g_prob = self.DC_L(code_rm)
        l_g = tf.reshape(tf.cast(tf.argmax(l_g_prob, axis=-1), dtype=tf.float32) * 0.2, shape=self.input_shape)

        # X_G,Y_G -> F_X_G,F_Y_G -> F_G_R
        f_x_g_r = self.get_f(x_g)
        f_y_g_r = self.get_f(y_g)

        # X_G -> L_X_G
        code_x_g = EC_X(x_g)
        l_g_prob_by_x = self.DC_L(code_x_g)
        l_g_by_x = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_G -> L_Y_G
        code_y_g = EC_Y(y_g)
        l_g_prob_by_y = self.DC_L(code_y_g)
        l_g_by_y = tf.reshape(tf.cast(tf.argmax(l_g_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # X_G -> Y_G_T
        y_g_t = DC_Y(self.SDC(code_x_g))
        # Y_G -> X_G_T
        x_g_t = DC_X(self.SDC(code_y_g))

        # D,FD
        j_x_g, j_x_g_c = self.D_M(x_g)
        j_y_g, j_y_g_c = self.D_M(y_g)
        j_code_rm = self.FD_R(code_rm)

        # 使得通过随机结构特征图生成的X模态图更逼真的对抗性损失
        D_X_loss = self.mse_loss(j_x_g, 0.0) * 25
        G_loss = self.mse_loss(j_x_g, 1.0) * 25

        # 使得通过随机结构特征图生成的Y模态图更逼真的对抗性损失
        D_Y_loss = self.mse_loss(j_y_g, 0.0) * 25
        G_loss += self.mse_loss(j_y_g, 1.0) * 25

        # 使得对随机结构特征图编码结果更加趋近于真实模态图编码结果的对抗性损失，
        # 以降低解码器解码难度，保证解码器能顺利解码出模态图
        D_F_loss = self.mse_loss(j_code_rm, 0.0) * 2
        G_loss += self.mse_loss(j_code_rm, 1.0) * 2

        # TODO 交叉熵损失函数
        D_X_loss += self.mse_loss(j_x_g_c, cx) * 25
        D_Y_loss += self.mse_loss(j_y_g_c, cy) * 25

        # 输入的结构特征图的重建自监督损失
        G_loss += self.mse_loss(f, f_x_g_r) * 20
        G_loss += self.mse_loss(f, f_y_g_r) * 20

        # 与输入的结构特征图融合后输入的肿瘤分割标签图的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成X模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_x[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_x[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_x[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_x[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_x[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_x[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g_by_x)

        # 与输入的结构特征图融合后输入的肿瘤分割标签图在生成Y模态后再次分割的重建自监督损失
        G_loss += self.mse_loss(label_expand[:, :, :, 0], l_g_prob_by_y[:, :, :, 0]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 1], l_g_prob_by_y[:, :, :, 1]) * 0.1 \
                  + self.mse_loss(label_expand[:, :, :, 2], l_g_prob_by_y[:, :, :, 2]) \
                  + self.mse_loss(label_expand[:, :, :, 3], l_g_prob_by_y[:, :, :, 3]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 4], l_g_prob_by_y[:, :, :, 4]) * 5 \
                  + self.mse_loss(label_expand[:, :, :, 5], l_g_prob_by_y[:, :, :, 5]) * 5
        G_loss += self.mse_loss(l, l_g_by_y)

        # 通过解码器生成X模态与Y模态图的编码 与 X模态与Y模态图经过编码器得到的编码 的自监督语义一致性损失
        G_loss += self.mse_loss(code_rm, code_x_g)
        G_loss += self.mse_loss(code_rm, code_y_g)
        G_loss += self.mse_loss(code_x_g, code_y_g) * 0.5

        # 生成的X模态与Y模态图进行转换得到的转换图与生成图的自监督损失
        G_loss += self.mse_loss(y_g, y_g_t) * 2
        G_loss += self.mse_loss(x_g, x_g_t) * 2

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_g * label_expand[0]) * 1.5
        G_loss += self.mse_loss(0.0, y_g * label_expand[0]) * 1.0
        G_loss += self.mse_loss(0.0, x_g_t * label_expand[0]) * 1.5
        G_loss += self.mse_loss(0.0, y_g_t * label_expand[0]) * 1.0

        image_list = [l,f, x_g, y_g, x_g_t, y_g_t,
                           l_g, l_g_by_x, l_g_by_y,
                           f_x_g_r, f_y_g_r]

        code_list = [code_rm, code_x_g, code_y_g]

        judge_list = [j_x_g, j_y_g, j_code_rm]

        return G_loss, D_X_loss,D_Y_loss,D_F_loss,image_list,code_list,judge_list

    def translate(self,x,y,cx,cy,l_x,l_y,EC_X, EC_Y, DC_X, DC_Y,G_loss=0.0,D_X_loss=0.0,D_Y_loss=0.0,D_F_loss=0.0,image_list=[],code_list=[],judge_list=[]):
        label_expand_x = tf.reshape(tf.one_hot(tf.cast(l_x, dtype=tf.int32), axis=-1, depth=6),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        label_expand_y = tf.reshape(tf.one_hot(tf.cast(l_y, dtype=tf.int32), axis=-1, depth=6),
                                    shape=[self.input_shape[0], self.input_shape[1], self.input_shape[2], 6])
        # X -> X_R
        code_x = EC_X(x)
        x_r = DC_X(self.SDC(code_x))
        # Y -> Y_R
        code_y = EC_Y(y)
        y_r = DC_Y(self.SDC(code_y))
        # X -> Y_T
        y_t = DC_Y(self.SDC(code_x))
        # Y -> X_T
        x_t = DC_X(self.SDC(code_y))
        # X -> L_X
        l_f_prob_by_x = self.DC_L(code_x)
        l_f_by_x = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_x, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y -> L_Y
        l_f_prob_by_y = self.DC_L(code_y)
        l_f_by_y = tf.reshape(tf.cast(tf.argmax(l_f_prob_by_y, axis=-1), dtype=tf.float32) * 0.2,
                              shape=self.input_shape)
        # Y_T -> X_C_R
        code_y_t = EC_Y(y_t)
        x_c_r = DC_X(self.SDC(code_y_t))

        # X_T -> Y_C_R
        code_x_t = EC_X(x_t)
        y_c_r = DC_Y(self.SDC(code_x_t))

        j_x, j_x_c = self.D_M(x)
        j_y, j_y_c = self.D_M(y)
        j_code_x = self.FD_R(code_x)
        j_code_y = self.FD_R(code_y)

        D_X_loss += self.mse_loss(j_x, 1.0) * 25
        D_Y_loss += self.mse_loss(j_y, 1.0) * 25
        D_F_loss += self.mse_loss(j_code_x, 1.0)
        D_F_loss += self.mse_loss(j_code_y, 1.0)

        # TODO 交叉熵损失函数
        D_X_loss += self.mse_loss(j_x_c, cx) * 25
        D_Y_loss += self.mse_loss(j_y_c, cy) * 25

        # X模态与Y模态图进行重建得到的重建图与原图的自监督损失
        G_loss += self.mse_loss(x, x_r) * 5
        G_loss += self.mse_loss(y, y_r) * 5

        # X模态与Y模态图进行转换得到的转换图与原图的有监督损失
        G_loss += self.mse_loss(x, x_c_r) * 10
        G_loss += self.mse_loss(y, y_c_r) * 10

        # X模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_x[:, :, :, 0], l_f_prob_by_x[:, :, :, 0]) \
                  + self.mse_loss(label_expand_x[:, :, :, 1], l_f_prob_by_x[:, :, :, 1]) \
                  + self.mse_loss(label_expand_x[:, :, :, 2], l_f_prob_by_x[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand_x[:, :, :, 3], l_f_prob_by_x[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 4], l_f_prob_by_x[:, :, :, 4]) * 15 \
                  + self.mse_loss(label_expand_x[:, :, :, 5], l_f_prob_by_x[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l_x, l_f_by_x) * 5

        # Y模态图分割训练的有监督损失
        G_loss += self.mse_loss(label_expand_y[:, :, :, 0], l_f_prob_by_y[:, :, :, 0]) \
                  + self.mse_loss(label_expand_y[:, :, :, 1], l_f_prob_by_y[:, :, :, 1]) \
                  + self.mse_loss(label_expand_y[:, :, :, 2], l_f_prob_by_y[:, :, :, 2]) * 5 \
                  + self.mse_loss(label_expand_y[:, :, :, 3], l_f_prob_by_y[:, :, :, 3]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 4], l_f_prob_by_y[:, :, :, 4]) * 15 \
                  + self.mse_loss(label_expand_y[:, :, :, 5], l_f_prob_by_y[:, :, :, 5]) * 15
        G_loss += self.mse_loss(l_y, l_f_by_y) * 5

        # X模态与Y模态图编码的有监督语义一致性损失
        G_loss += self.mse_loss(code_x, code_y_t) * 5
        G_loss += self.mse_loss(code_y, code_x_t) * 5

        # 限制像素生成范围为脑主体掩膜的范围的监督损失
        G_loss += self.mse_loss(0.0, x_r * label_expand_x[0]) * 0.5
        G_loss += self.mse_loss(0.0, y_r * label_expand_y[0]) * 0.5
        G_loss += self.mse_loss(0.0, x_t * label_expand_y[0]) * 0.5
        G_loss += self.mse_loss(0.0, y_t * label_expand_x[0]) * 0.5

        image_list.extend([x,y,x_r, y_r, x_t, y_t,
                                l_f_by_x, l_f_by_y])

        code_list.extend([code_x, code_y])

        judge_list.extend([j_x, j_x, j_code_x, j_code_y])

        return  G_loss, D_X_loss,D_Y_loss,D_F_loss,image_list,code_list,judge_list

    def model(self,x, y,cx,cy,l,f,EC_X, EC_Y, DC_X, DC_Y):
        G_loss,D_X_loss,D_Y_loss,D_F_loss,image_list,code_list,judge_list = self.gen(f, l,cx,cy, EC_X, EC_Y, DC_X, DC_Y)
        G_loss,D_X_loss,D_Y_loss,D_F_loss,image_list,code_list,judge_list= self.translate(x, y,cx,cy, l,l, EC_X, EC_Y, DC_X, DC_Y, G_loss, D_X_loss,D_Y_loss, D_F_loss,image_list,code_list,judge_list)
        return G_loss, D_X_loss+D_Y_loss+D_F_loss,image_list,code_list,judge_list

    def run(self, x, y, z, w, l, rand_f,r):
        # X,Y -> F
        # 选择f来源模态
        m = tf.case({tf.equal(rand_f, 0): lambda: x,
                     tf.equal(rand_f, 1): lambda: y,
                     tf.equal(rand_f, 2): lambda: z,
                     tf.equal(rand_f, 3): lambda: w}, exclusive=True)
        f = self.get_f(m)  # M -> F

        G_loss_XY, D_loss_XY,image_list_XY,code_list_XY,judge_list_XY = self.model( x, y, 0., 1.,l, f, self.EC_X, self.EC_Y, self.DC_X, self.DC_Y)

        G_loss_XZ, D_loss_XZ,image_list_XZ,code_list_XZ,judge_list_XZ = self.model( x, z, 0., 2.,l, f, self.EC_X, self.EC_Z, self.DC_X, self.DC_Z)

        G_loss_XW, D_loss_XW,image_list_XW,code_list_XW,judge_list_XW = self.model( x, w, 0., 3.,l, f, self.EC_X, self.EC_W, self.DC_X, self.DC_W)

        G_loss_YZ, D_loss_YZ,image_list_YZ,code_list_YZ,judge_list_YZ = self.model( y, z, 1., 2.,l, f, self.EC_Y, self.EC_Z, self.DC_Y, self.DC_Z)

        G_loss_YW, D_loss_YW,image_list_YW,code_list_YW,judge_list_YW = self.model( y, w, 1., 3.,l, f, self.EC_Y, self.EC_W, self.DC_Y, self.DC_W)

        G_loss_ZW, D_loss_ZW,image_list_ZW,code_list_ZW,judge_list_ZW = self.model(z, w, 2., 3., l, f,self.EC_Z, self.EC_W, self.DC_Z, self.DC_W)

        self.image_list = [image_list_XY,image_list_XZ,image_list_XW,image_list_YZ,image_list_YW,image_list_ZW]
        self.code_list = [code_list_XY,code_list_XZ,code_list_XW,code_list_YZ,code_list_YW,code_list_ZW]
        self.judge_list = [judge_list_XY,judge_list_XZ,judge_list_XW,judge_list_YZ,judge_list_YW,judge_list_ZW]

        loss_list = [G_loss_XY+G_loss_XZ+G_loss_XW+G_loss_YZ+G_loss_YW+G_loss_ZW,
                     D_loss_XY+D_loss_XZ+D_loss_XW+D_loss_YZ+D_loss_YW+D_loss_ZW]

        return  loss_list

    def get_variables(self):
        return [self.EC_R.variables
                + self.DC_L.variables
                + self.EC_X.variables
                + self.DC_X.variables
                +self.EC_Y.variables
                + self.DC_Y.variables
                +self.EC_Z.variables
                + self.DC_Z.variables
                +self.EC_W.variables
                + self.DC_W.variables
                + self.SDC.variables
            ,
                self.D_M.variables+
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
        j_x, j_x_g, j_y, j_y_g, j_code_x, j_code_y, j_code_rm = \
            j_list[0], j_list[1], j_list[2], j_list[3], j_list[4], j_list[5], j_list[6]
        tf.summary.histogram('discriminator/TRUE/j_x', j_x)
        tf.summary.histogram('discriminator/TRUE/j_y', j_y)
        tf.summary.histogram('discriminator/TRUE/j_code_x', j_code_x)
        tf.summary.histogram('discriminator/TRUE/j_code_y', j_code_y)

        tf.summary.histogram('discriminator/FALSE/j_x_g', j_x_g)
        tf.summary.histogram('discriminator/FALSE/j_y_g', j_y_g)
        tf.summary.histogram('discriminator/FALSE/j_code_rm', j_code_rm)

    def loss_summary(self, loss_list):
        G_loss, D_loss = loss_list[0], loss_list[1]
        tf.summary.scalar('loss/G_loss', G_loss)
        tf.summary.scalar('loss/D_loss', D_loss)

    def evaluation_code(self, code_list):
        code_rm, code_x_g, code_y_g, \
        code_x, code_y = \
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
        x, y, f, l, \
        x_g, y_g, x_g_t, y_g_t, l_g, l_g_by_x, l_g_by_y, f_x_g_r, f_y_g_r, \
        x_r, y_r, x_t, y_t, x_c_r, y_c_r, l_f_by_x, l_f_by_y = \
            image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], \
            image_list[6], image_list[7], image_list[8], image_list[9], image_list[10], image_list[11], \
            image_list[12], image_list[13], image_list[14], image_list[15], image_list[16], image_list[17], \
            image_list[18], image_list[19], image_list[20]

        list = [self.PSNR(x, x_t), self.PSNR(x, x_r),
                self.PSNR(y, y_t), self.PSNR(y, y_r),
                self.PSNR(x_g, x_g_t),
                self.PSNR(y_g, y_g_t),
                self.PSNR(l, l_f_by_x), self.PSNR(l, l_f_by_y),
                self.PSNR(l, l_g), self.PSNR(l, l_g_by_x), self.PSNR(l, l_g_by_y),
                self.PSNR(f, f_x_g_r), self.PSNR(f, f_y_g_r),

                self.SSIM(x, x_t), self.SSIM(x, x_r),
                self.SSIM(y, y_t), self.SSIM(y, y_r),
                self.SSIM(x_g, x_g_t),
                self.SSIM(y_g, y_g_t),
                self.SSIM(l, l_f_by_x), self.SSIM(l, l_f_by_y),
                self.SSIM(l, l_g), self.SSIM(l, l_g_by_x), self.SSIM(l, l_g_by_y),
                self.SSIM(f, f_x_g_r), self.SSIM(f, f_y_g_r),
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
        tf.summary.scalar('evaluation/PSNR/f__VS__f_xy_g_r', evluation_list[11])
        tf.summary.scalar('evaluation/PSNR/f__VS__f_xy_g_r', evluation_list[12])

        tf.summary.scalar('evaluation/SSIM/x__VS__x_t', evluation_list[13])
        tf.summary.scalar('evaluation/SSIM/x__VS__x_r', evluation_list[14])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_t', evluation_list[15])
        tf.summary.scalar('evaluation/SSIM/y__VS__y_r', evluation_list[16])
        tf.summary.scalar('evaluation/SSIM/x_g__VS__x_g_t', evluation_list[17])
        tf.summary.scalar('evaluation/SSIM/y_g__VS__y_g_t', evluation_list[18])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_x', evluation_list[19])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_f_by_y', evluation_list[20])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g', evluation_list[21])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_x', evluation_list[22])
        tf.summary.scalar('evaluation/SSIM/l_input__VS__l_g_by_y', evluation_list[23])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_xy_g_r', evluation_list[24])
        tf.summary.scalar('evaluation/SSIM/f__VS__f_xy_g_r', evluation_list[25])

    def image_summary(self, image_list):
        x, y, f, l, \
        x_g, y_g, x_g_t, y_g_t, l_g, l_g_by_x, l_g_by_y, f_x_g_r, f_y_g_r, \
        x_r, y_r, x_t, y_t, x_c_r, y_c_r, l_f_by_x, l_f_by_y = \
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

        tf.summary.image('image/l_input', l)
        tf.summary.image('image/l_g', l_g)
        tf.summary.image('image/l_f_by_x', l_f_by_x)
        tf.summary.image('image/l_f_by_y', l_f_by_y)
        tf.summary.image('image/l_g_by_x', l_g_by_x)
        tf.summary.image('image/l_g_by_y', l_g_by_y)

        tf.summary.image('image/f', f)
        tf.summary.image('image/f_x_g_r', f_x_g_r)
        tf.summary.image('image/f_y_g_r', f_y_g_r)

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

