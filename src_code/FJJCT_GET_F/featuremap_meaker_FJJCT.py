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
SAVE_F3 = "D:/BaiduYunDownload/TC19/new_images_F3"
SAVE_M3 = "D:/BaiduYunDownload/TC19/new_images_M3"
SAVE_F = "D:/BaiduYunDownload/TC19/new_images_F"
SAVE_M = "D:/BaiduYunDownload/TC19/new_images_M"
NUM = "318818_002"
alpha=0.01
k_size1=5
p=2
beta=0.48
k_size2=3


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

    fx = get_f(x, j=alpha)
    mask_x = get_mask(x, p=2, beta=beta)


with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    try:
        os.makedirs(SAVE_F1)
        os.makedirs(SAVE_M1)
        os.makedirs(SAVE_F2)
        os.makedirs(SAVE_M2)
        os.makedirs(SAVE_F3)
        os.makedirs(SAVE_M3)
        os.makedirs(SAVE_F)
        os.makedirs(SAVE_M)
    except os.error:
        pass

    files = os.listdir(PATH)
    for i in range(len(files)):
        file=files[i]
        input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + file))
        input_x = np.asarray(input_x).reshape([512, 512, 3])

        fx1_,mask_x1_ = sess.run([fx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 0:1]]).astype( 'float32') })
        fx1_ = signal.medfilt2d(np.asarray(fx1_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x1_ = signal.medfilt2d(np.asarray(mask_x1_)[0, :, :, 0, ], kernel_size=k_size2)

        fx2_, mask_x2_ = sess.run([fx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 1:2]]).astype('float32')})
        fx2_ = signal.medfilt2d(np.asarray(fx2_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x2_ = signal.medfilt2d(np.asarray(mask_x2_)[0, :, :, 0, ], kernel_size=k_size2)

        fx3_, mask_x3_ = sess.run([fx, mask_x], feed_dict={x: np.asarray([input_x[:, :, 2:3]]).astype('float32')})
        fx3_ = signal.medfilt2d(np.asarray(fx3_)[0, :, :, 0, ], kernel_size=k_size1)
        mask_x3_ = signal.medfilt2d(np.asarray(mask_x3_)[0, :, :, 0, ], kernel_size=k_size2)

        fx_=np.zeros([512, 512, 3])
        mask_x_ = np.zeros([512, 512, 3])
        fx_[:, :, 0] = (1.0 - mask_x1_) * fx1_
        fx_[:, :, 1] = (1.0 - mask_x2_) * fx2_
        fx_[:, :, 2] = (1.0 - mask_x3_) * fx3_
        mask_x_[:, :, 0] = mask_x1_
        mask_x_[:, :, 1] = mask_x2_
        mask_x_[:, :, 2] = mask_x3_

        NUM=file.replace(".mha", "")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x1_) * fx1_), SAVE_F1 + "/" + NUM+ ".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x1_), SAVE_M1 + "/" +NUM+ ".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x2_) * fx2_), SAVE_F2 + "/" + NUM+".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x2_), SAVE_M2 + "/" + NUM+ ".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x3_) * fx3_), SAVE_F3 + "/" + NUM+".tiff")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x3_), SAVE_M3 + "/" + NUM+".tiff")

        SimpleITK.WriteImage(SimpleITK.GetImageFromArray((1.0 - mask_x_) * fx_), SAVE_F + "/" + NUM+".mha")
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_x_), SAVE_M + "/" + NUM+ ".mha")

