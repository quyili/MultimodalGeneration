# _*_ coding:utf-8 _*_
import tensorflow as tf
from detector import Detector


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
        self.tenaor_name = {}

        self.LESP = Detector('LESP', ngf=ngf, keep_prob=0.99)

    def model(self,input,groundtruth_class,groundtruth_location,groundtruth_positives,groundtruth_negatives):
        feature_class,feature_location,feature_maps_shape = self.LESP(input)

        # 损失函数
        self.groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
        self.softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                                    labels=groundtruth_class)
        self.loss_location = tf.div(tf.reduce_sum(tf.multiply(
            tf.reduce_sum(self.smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                          reduction_indices=2), groundtruth_positives), reduction_indices=1),
                                    tf.reduce_sum(groundtruth_positives, reduction_indices=1))
        self.loss_class = tf.div(
            tf.reduce_sum(tf.multiply(self.softmax_cross_entropy, self.groundtruth_count), reduction_indices=1),
            tf.reduce_sum(self.groundtruth_count, reduction_indices=1))
        self.loss_all = tf.reduce_sum(tf.add(self.loss_class, self.loss_location))

        return  [self.loss_all]

    def get_variables(self):
        return self.LESP.variables

    def optimize(self):
        def make_optimizer(name='Adam'):
            learning_step = (
                tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, name=name)
            )
            return learning_step
        D_optimizer = make_optimizer(name='Adam_D')

        return  D_optimizer

    def loss_summary(self, L_loss):
        tf.summary.scalar('loss/L_loss', L_loss[0])

    def acc(self,x,y):
         correct_prediction = tf.equal(x, y)
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         return accuracy 

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

   # smooth_L1 算法
    def smooth_L1(self, x):
        return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)),
                        tf.subtract(tf.abs(x), 0.5))