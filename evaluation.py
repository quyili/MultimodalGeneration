# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import metrics.sliced_wasserstein as SWD
import metrics.inception_score as IS
import metrics.frechet_inception_distance as FID

PATH = "D:/BaiduYunDownload/finding-lungs-in-ct-data/2d_images"
NUM = "ID_0264_Z_0080"

def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output

def MSSSIM(output, target):
    ssim = tf.reduce_mean(tf.image.ssim_multiscale(output, target, max_val=1.0))
    return ssim

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1,512, 512, 1])
    y = tf.placeholder(tf.float32, shape=[1,512, 512, 1])
    out1=MSSSIM(x, y)

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + NUM + ".tif"))
    input_y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + NUM + ".tif"))

    out2_ =  SWD.sliced_wasserstein(input_x, input_y, 4, 128)

    out3_ = IS.inception_score("D:/BaiduYunDownload/SWM/test/X","D:\\project\\MultimodalGeneration\\metrics")
    out4_ = FID.calculate_fid_given_paths(["D:/BaiduYunDownload/SWM/test/X","D:/BaiduYunDownload/SWM/test/X"],"D:/project/MultimodalGeneration/metrics")

    out1_ = sess.run(out1,feed_dict={x: np.asarray([input_x.reshape([512, 512, 1])]).astype('float32'),
                                     y: np.asarray([input_y.reshape([512, 512, 1])]).astype('float32')})
    print(out1_,out2_,out3_,out4_)
