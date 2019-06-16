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
    y = tf.placeholder(tf.float32, shape=[1, 184, 144, 1])
    x1 = norm(tf.reduce_mean(tf.image.sobel_edges(x), axis=-1))
    x2 = norm(tf.reduce_max(tf.image.sobel_edges(x), axis=-1))

    y1 = norm(tf.reduce_mean(tf.image.sobel_edges(y), axis=-1))
    y2 = norm(tf.reduce_max(tf.image.sobel_edges(y), axis=-1))

    xy1 = norm(tf.reduce_mean(tf.concat([x1,y1],axis=-1), axis=-1,keep_dims=True))
    xy2 =norm(tf.reduce_max(tf.concat([x2,y2],axis=-1), axis=-1,keep_dims=True))

    out = xy2-tf.reduce_mean(xy2, axis=[1, 2, 3])
    out =  tf.ones(out.get_shape().as_list())*tf.cast( out>0.07,dtype="float32")


with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    input_x = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT1/0_90.tiff")).astype('float32')
    input_y = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage("../mydata/BRATS2015/testT2/0_90.tiff")).astype('float32')
    input_x = np.asarray(input_x).reshape([184, 144, 1])
    input_y = np.asarray(input_y).reshape([184, 144, 1])
    x1_,x2_,y1_,y2_,xy1_,xy2_,out_ = sess.run([x1,x2,y1,y2,xy1,xy2,out],
                   feed_dict={x: np.asarray([input_x]),
                              y: np.asarray([input_y])})

    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x1_)[0, :, :, 0]), "x1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(x2_)[0, :, :, 0]), "x2_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y1_)[0, :, :, 0]), "y1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(y2_)[0, :, :, 0]), "y2_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(xy1_)[0, :, :, 0]), "xy1_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(xy2_)[0, :, :, 0]), "xy2_.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_x[:, :, 0]), "input_x.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(input_y[:, :, 0]), "input_y.tiff")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(np.asarray(out_)[0, :, :, 0]), "out.tiff")
