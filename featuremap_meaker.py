# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK


def norm(input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    y = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])

    x1 = norm(tf.reduce_min(tf.image.sobel_edges(x), axis=-1))
    x2 = norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))

    x1 = tf.reduce_mean(x1, axis=[1, 2, 3])-x1
    x2 = x2 - tf.reduce_mean(x2, axis=[1, 2, 3])

    x1 = tf.ones(x1.get_shape().as_list()) * tf.cast(x1 > 0.15, dtype="float32")
    x2 = tf.ones(x2.get_shape().as_list()) * tf.cast(x2 > 0.15, dtype="float32")

    x12 = x1 + x2
    x12 = tf.ones(x12.get_shape().as_list()) * tf.cast(x12 > 0.0, dtype="float32")



    y1 = norm(tf.reduce_min(tf.image.sobel_edges(y), axis=-1))
    y2 = norm(tf.reduce_max(tf.image.sobel_edges(y), axis=-1))

    y1 = tf.reduce_mean(y1, axis=[1, 2, 3])- y1
    y1 = tf.ones(y1.get_shape().as_list()) * tf.cast(y1 > 0.1, dtype="float32")

    y2 = y2- tf.reduce_mean(y2, axis=[1, 2, 3])
    y2 = tf.ones(y2.get_shape().as_list()) * tf.cast(y2 > 0.1, dtype="float32")

    y12 = y1 + y2
    y12 = tf.ones(y12.get_shape().as_list()) * tf.cast(y12 > 0.0, dtype="float32")

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1/0_90.tiff")).astype('float32')
    input_y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT2/0_90.tiff")).astype('float32')
    input_x = np.asarray(input_x).reshape([184, 144, 1])
    input_y = np.asarray(input_y).reshape([184, 144, 1])
    x12_, y12_,x1_, x2_, y1_, y2_ = sess.run([x12, y12, x1, x2, y1, y2],
                                                    feed_dict={x: np.asarray([input_x]),
                                                               y: np.asarray([input_y])})

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x1_)[0, :, :, 0]), "x1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x2_)[0, :, :, 0]), "x2_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y12_)[0, :, :, 0]), "y12_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x12_)[0, :, :, 0]), "x12_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y1_)[0, :, :, 0]), "y1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y2_)[0, :, :, 0]), "y2_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x[:, :, 0]), "input_x.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_y[:, :, 0]), "input_y.tiff")

