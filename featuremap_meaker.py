# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    y = tf.image.sobel_edges(x)

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1/0_90.tiff")).astype('float32')
    input_x = np.asarray(input_x).reshape([184, 144, 1])
    out = sess.run(y,
                   feed_dict={x: np.asarray([input_x])})
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x[:, :, 0]), "input_x.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, 0, 0]), "ouput_0.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, 0, 1]), "ouput_1.tiff")
    SimpleITK.WriteImage(
        SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, 0, 0] * 0.5 + np.asarray(out)[0, :, :, 0, 1] * 0.5),
        "ouput_mean.tiff")
    SimpleITK.WriteImage(
        SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, 0, 0] + np.asarray(out)[0, :, :, 0, 1]),
        "ouput_sum.tiff")
