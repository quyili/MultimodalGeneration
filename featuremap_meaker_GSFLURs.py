# -*- coding: gbk -*-
import tensorflow as tf
import numpy as np
import SimpleITK
import cv2
import scipy.signal as signal
from skimage import transform
import os

PATH = "E:/project/MultimodalGeneration/data/chest_xray/test/F_"
SAVE_F = "E:/project/MultimodalGeneration/data/chest_xray/test/F2_"
# sigma=0.4
# alpha=0.2
beta=1

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

def gaussian_blur_op(image, kernel, kernel_size, cdim=3):
    # kernel as placeholder variable, so it can change
    outputs = []
    pad_w = (kernel_size*kernel_size - 1) // 2
    padded = tf.pad(image, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
    for channel_idx in range(cdim):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        g = tf.reshape(kernel, [1, -1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        g = tf.reshape(kernel, [-1, 1, 1, 1])
        data_c = tf.nn.conv2d(data_c, g, [1, 1, 1, 1], 'VALID')
        outputs.append(data_c)
    return tf.concat(outputs, axis=3)

def gaussian_blur(x,sigma=0.5,alpha=0.15):
    gauss_filter = gauss_2d_kernel(3, sigma)
    gauss_filter = gauss_filter.astype(dtype=np.float32)
    y = gaussian_blur_op(x, gauss_filter, 3, cdim=1)
    y = tf.ones(y.get_shape().as_list()) * tf.cast(y > alpha, dtype="float32")
    return y

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

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 512, 512, 1])
    # x1=x[:,:,:,0:1]
    # x2=x[:,:,:,1:2]
    # x3=x[:,:,:,2:3]
    #
    # y1 = gaussian_blur(x1, sigma=0.5, alpha=0.3)
    # y2 = gaussian_blur(x2, sigma=0.5, alpha=0.3)
    # y3 = gaussian_blur(x3, sigma=0.5, alpha=0.3)
    #
    # y=tf.concat([y1,y2,y3],axis=-1)

    y=x

    # y =gaussian_blur(y,sigma=0.5, alpha=0.15)

    # y = get_f(y, j=0.25)
    #
    # y = gaussian_blur(y, sigma=0.7, alpha=0.3)

    y = gaussian_blur(y,sigma=0.01,alpha=0.005)



with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    try:
        os.makedirs(SAVE_F)
    except os.error:
        pass

    files = os.listdir(PATH)
    for i in range(len(files)):
        file = files[i]
        input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(PATH + "/" + file)).astype('float32')
        input_x = transform.resize(np.asarray(input_x), [512, 512, 1])
        y_ = sess.run(y,feed_dict={x: np.asarray([input_x]) })
        y_ = np.asarray(y_)[0, :, :, 0]
        y_ = signal.medfilt2d( y_ , kernel_size=beta)
        # print(np.asarray(y_).shape)
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray( y_ ), SAVE_F + "/" + file )
