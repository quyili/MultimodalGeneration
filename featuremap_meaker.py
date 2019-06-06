# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import SimpleITK


def norm( input):
    output = (input - tf.reduce_min(input, axis=[1, 2, 3])
              ) / (tf.reduce_max(input, axis=[1, 2, 3]) - tf.reduce_min(input, axis=[1, 2, 3]))
    return output


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    y = norm(tf.reduce_mean(tf.image.sobel_edges(x), axis=-1))

with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1/0_90.tiff")).astype('float32')
    input_x = np.asarray(input_x).reshape([184, 144, 1])
    out = sess.run(y,
                   feed_dict={x: np.asarray([input_x])})
    print(np.asarray(out).shape)

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, :]), "ouput_mean.mha")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out)[0, :, :, 0]), "ouput_mean.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x[:, :, 0]), "input_x.tiff")
