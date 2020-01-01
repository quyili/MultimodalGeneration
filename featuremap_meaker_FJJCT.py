# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal
from skimage import transform
import os

PATH = "D:/BaiduYunDownload/TC19/new_images"
SAVE_F1 = "D:/BaiduYunDownload/TC19/new_images_F1"
SAVE_M1 = "D:/BaiduYunDownload/TC19/new_images_M1"
SAVE_F2 = "D:/BaiduYunDownload/TC19/new_images_F2"
SAVE_M2 = "D:/BaiduYunDownload/TC19/new_images_M2"
NUM = "318818_002"
alpha=0.01
k_size1=5
# alpha=0.04
# k_size1=7

p=2
beta=0.48
k_size2=3
# beta=0.22
# k_size2=5

IF_SAVE_F=True
IF_SAVE_M=True


def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output

def get_f(x, j=0.1):
    x1 = norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
    x2 = norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))

    x1 = tf.reduce_mean(x1, axis=[1, 2, 3]) - x1
    x2 = x2 - tf.reduce_mean(x2, axis=[1, 2, 3])

    x1 = tf.ones(x1.get_shape().as_list()) * tf.cast(x1 > j, dtype="float32")
    x2 = tf.ones(x2.get_shape().as_list()) * tf.cast(x2 > j, dtype="float32")

    x12 = x1 + x2
    x12 = tf.ones(x12.get_shape().as_list()) * tf.cast(x12 > 0.0, dtype="float32")
    return x12


def get_mask(m, p=5,beta=0.0):
    m=norm(m)
    mask = 1.0 - tf.ones(m.get_shape().as_list()) * tf.cast(m > beta, dtype="float32")
    shape = m.get_shape().as_list()
    mask = tf.image.resize_images(mask, size=[shape[1] + p, shape[2] + p], method=1)
    mask = tf.image.resize_image_with_crop_or_pad(mask, shape[1], shape[2])
    return mask


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 512, 512, 1])

    ## 3D
    # fx = get_f(x, j=0.09)
    # mask_x = get_mask(x, p=2, beta=0.2)

    fx = get_f(x, j=alpha)
    mask_x = get_mask(x, p=p, beta=beta)


with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + NUM + ".mha"))
    input_x = input_x.reshape([512, 512, 3])
    fx_, mask_x_ = sess.run([fx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 0:1]]).astype('float32')})
    fx_ = signal.medfilt2d(np.asarray(fx_)[0, :, :, 0, ], kernel_size=k_size1)
    mask_x_ = signal.medfilt2d(np.asarray(mask_x_)[0, :, :, 0, ], kernel_size=k_size2)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(fx_), "X0.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x_), "X1.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x_) * fx_),  "X2.tiff")
    # if IF_SAVE_M==True:
    #     SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x_), SAVE_M + "/" + NUM + ".tiff")
    # if IF_SAVE_F==True:
    #     SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x_) * fx_), SAVE_F + "/" + NUM + ".tiff")
    #     # SimpleITK.WriteImage(SimpleITK.GetImageFromArray(fx_), SAVE_F + "/" + NUM + ".tiff")
