# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal


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
    x = tf.placeholder(tf.float32, shape=[1, 584, 565, 3])

    fx = get_f(x, j=0.03)
    mask_x = get_mask(x, p=10, beta=0.3)
    out_x = (1.0-mask_x) * fx

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("D:/BaiduYunDownload/drive/training/images/28_training.tif")).astype(
        'float32')
    input_x = np.asarray(input_x).reshape([584, 565, 3])
    fx_,mask_x_,out_x_ = sess.run([fx, mask_x, out_x],
                                                           feed_dict={x: np.asarray([input_x]) })
    fx_ = signal.medfilt2d(np.asarray(fx_)[0, :, :, 0, ], kernel_size=1)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(fx_), "X0_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(mask_x_)[0, :, :, 0]), "X1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out_x_)[0, :, :, 0]), "X2_.tiff")
