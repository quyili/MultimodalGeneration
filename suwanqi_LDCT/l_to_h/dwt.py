# _*_ coding:utf-8 _*_
import tensorflow as tf
import pywt
import numpy as np


def tf_dwt(yl, wave='haar'):
    w = pywt.Wavelet(wave)
    ll = np.outer(w.dec_lo, w.dec_lo)
    lh = np.outer(w.dec_hi, w.dec_lo)
    hl = np.outer(w.dec_lo, w.dec_hi)
    hh = np.outer(w.dec_hi, w.dec_hi)
    d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
    d_temp[::-1, ::-1, 0, 0] = ll
    d_temp[::-1, ::-1, 0, 1] = lh
    d_temp[::-1, ::-1, 0, 2] = hl
    d_temp[::-1, ::-1, 0, 3] = hh
    filts = d_temp.astype('float32')

    filts = np.copy(filts)

    filter = tf.convert_to_tensor(filts)
    sz = 2 * (len(w.dec_lo) // 2 - 1)

    with tf.variable_scope('DWT'):
        yl = tf.pad(yl, tf.constant([[0, 0], [sz, sz], [sz, sz], [0, 0]]), mode='reflect')
        outputs = tf.nn.conv2d(yl[:, :, :, 0:1], filter, padding='VALID', strides=[1, 2, 2, 1])

    return outputs

# def tf_idwt(y,  wave='haar'):
#     w = pywt.Wavelet(wave)
#     ll = np.outer(w.rec_lo, w.rec_lo)
#     lh = np.outer(w.rec_hi, w.rec_lo)
#     hl = np.outer(w.rec_lo, w.rec_hi)
#     hh = np.outer(w.rec_hi, w.rec_hi)
#     d_temp = np.zeros((np.shape(ll)[0], np.shape(ll)[1], 1, 4))
#     d_temp[:, :, 0, 0] = ll
#     d_temp[:, :, 0, 1] = lh
#     d_temp[:, :, 0, 2] = hl
#     d_temp[:, :, 0, 3] = hh
#     filts = d_temp.astype('float32')
#     filts = filts[None, :, :, :, :]
#     filter = tf.convert_to_tensor(filts)
#     s = 2 * (len(w.dec_lo) // 2 - 1)
#     out_size = tf.shape(y)[1]
#
#     with tf.variable_scope('IWT'):
#         y = tf.expand_dims(y, 1)
#         inputs = tf.split(y, [4] * int(int(y.shape.dims[4])/4), 4)
#         inputs = tf.concat([x for x in inputs], 1)
#
#         outputs_3d = tf.nn.conv3d_transpose(inputs, filter, output_shape=[tf.shape(y)[0], tf.shape(inputs)[1],
#                                                                           2*(out_size-1)+np.shape(ll)[0],
#                                                                           2*(out_size-1)+np.shape(ll)[0], 1],
#                                             padding='VALID', strides=[1, 1, 2, 2, 1])
#         outputs = tf.split(outputs_3d, [1] * int(int(y.shape.dims[4])/4), 1)
#         outputs = tf.concat([x for x in outputs], 4)
#
#         outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[2],
#                                        tf.shape(outputs)[3], tf.shape(outputs)[4]))
#         outputs = outputs[:, s: 2 * (out_size - 1) + np.shape(ll)[0] - s, s: 2 * (out_size - 1) + np.shape(ll)[0] - s,
#                   :]
#     return outputs
