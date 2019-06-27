# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK


def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output

def get_f(x,j=0.1):
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
    x = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    y = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    z = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    w = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])

    fx = get_f(x, j=0.12)
    fy = get_f(y, j=0.12)
    fz = get_f(z, j=0.12)
    fw = get_f(w, j=0.12)

    # fx = get_f(x, j=0.07)
    # fy = get_f(y, j=0.07)
    # fz = get_f(z, j=0.07)
    # fw = get_f(w, j=0.07)


with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1/0_90.tiff")).astype('float32')
    input_y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT2/0_90.tiff")).astype('float32')
    input_z = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1c/0_90.tiff")).astype('float32')
    input_w = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testFlair/0_90.tiff")).astype('float32')
    input_x = np.asarray(input_x).reshape([184, 144, 1])
    input_y = np.asarray(input_y).reshape([184, 144, 1])
    input_z = np.asarray(input_z).reshape([184, 144, 1])
    input_w = np.asarray(input_w).reshape([184, 144, 1])
    fx_, fy_,fz_, fw_= sess.run([fx,fy,fz,fw],feed_dict={x: np.asarray([input_x]),
                                                       y: np.asarray([input_y]),
                                                       z: np.asarray([input_z]),
                                                       w: np.asarray([input_w])
                                                       })


    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(fx_)[0, :, :, 0]), "fx_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(fy_)[0, :, :, 0]), "fy_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(fz_)[0, :, :, 0]), "fz_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(fw_)[0, :, :, 0]), "fw_.tiff")

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x[:, :, 0]), "input_x.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_y[:, :, 0]), "input_y.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_z[:, :, 0]), "input_z.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_w[:, :, 0]), "input_w.tiff")

