# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal
from skimage import transform

PATH = "E:/project/MultimodalGeneration/data/SWM/train/F"
SAVE_F = "E:/project/MultimodalGeneration/data/SWM/train/NEW_F"
NUM = "21_training"
sigma = 0.7
alpha = 0.3
beta = 1


def gauss_2d_kernel(kernel_size=3, sigma=0.0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = (kernel_size - 1) / 2
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val


def gaussian_blur(image, kernel, kernel_size, cdim=3):
    # kernel as placeholder variable, so it can change
    outputs = []
    pad_w = (kernel_size * kernel_size - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = tf.reshape(kernel, [1, -1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = tf.reshape(kernel, [-1, 1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)


def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 512, 512, 1])

    gauss_filter = gauss_2d_kernel(3, sigma)
    gauss_filter = gauss_filter.astype(dtype=np.float32)
    x = gaussian_blur(x, gauss_filter, 3)

    y = tf.ones(x.get_shape().as_list()) * tf.cast(x > alpha, dtype="float32")

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(
        SimpleITK.ReadImage(PATH + "/" + NUM + ".tif")).astype(
        'float32')
    input_x = transform.resize(np.asarray(input_x), [512, 512, 1])
    y_ = sess.run(y, feed_dict={x: np.asarray([input_x])})
    y_ = signal.medfilt2d(np.asarray(y_)[0, :, :, 0], kernel_size=beta)
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(y_), SAVE_F + "/" + NUM + ".tiff")
