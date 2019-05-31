# _*_ coding:utf-8 _*_
import tensorflow as tf


def relu(x, alpha=0.2, max_value=100.0):
    '''
    leaky relu
    alpha: slope of negative section.
    '''
    x = tf.maximum(alpha * x, x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))
    return x


def uk_resize(input, reuse=False, name=None, output_size=None):
    """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
        with k filters, stride 1/2
    Args:
      input: 4D tensor
      k: integer, number of filters (output depth)
      norm: 'instance' or 'batch' or None
      is_training: boolean or BoolTensor
      reuse: boolean
      name: string, e.g. 'c7sk-32'
      output_size: integer, desired output size of layer
    Returns:
      4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('resize', reuse=reuse):
            input_shape = input.get_shape().as_list()
            if not output_size:
                output_size = [input_shape[1] * 2, input_shape[2] * 2]
            up_sample = tf.image.resize_images(input, output_size, method=1)
        return up_sample


def uk_resize_3d(input, reuse=False, name=None, output_size=None):
    """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
        with k filters, stride 1/2
    Args:
      input: 4D tensor
      k: integer, number of filters (output depth)
      norm: 'instance' or 'batch' or None
      is_training: boolean or BoolTensor
      reuse: boolean
      name: string, e.g. 'c7sk-32'
      output_size: integer, desired output size of layer
    Returns:
      4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('resize', reuse=reuse):
            input_shape = input.get_shape().as_list()
            if not output_size:
                output_size = [input_shape[1] * 2, input_shape[2] * 2, input_shape[3] * 2]
            up_sample = []
            for i in range(input_shape[0]):
                input_channl = input[i, :, :, :, :]
                up_sample_channl = tf.image.resize_images(input_channl, [output_size[1], output_size[2]], method=1)
                up_sample_channl = tf.transpose(up_sample_channl, perm=[1, 0, 2, 3])
                up_sample_channl = tf.image.resize_images(up_sample_channl, [output_size[0], output_size[2]],
                                                          method=1)
                up_sample.append(up_sample_channl)
            up_sample = tf.concat(up_sample, 0)
            up_sample = tf.reshape(up_sample, [-1, output_size[1], output_size[0], output_size[2], input_shape[4]])
            up_sample = tf.transpose(up_sample, perm=[0, 2, 1, 3, 4])
        return up_sample


### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
    """ Helper to create an initialized Variable
    Args:
      name: name of the variable
      shape: list of ints
      mean: mean of a Gaussian
      stddev: standard deviation of a Gaussian
    Returns:
      A trainable variable
    """

    var = tf.get_variable(
        name, shape,
        initializer=tf.random_normal_initializer(
            mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant))


def _norm(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalization or None
    """
    if norm == 'instance':
        return _instance_norm(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def _norm_3d(input, is_training, norm='instance'):
    """ Use Instance Normalization or Batch Normalization or None
    """
    if norm == 'instance':
        return _instance_norm_3d(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input


def _batch_norm(input, is_training):
    """ Batch Normalization
    """
    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(input,
                                            decay=0.9,
                                            scale=True,
                                            updates_collections=None,
                                            is_training=is_training)


def _instance_norm(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[3]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def _instance_norm_3d(input):
    """ Instance Normalization
    """
    with tf.variable_scope("instance_norm"):
        depth = input.get_shape()[4]
        scale = _weights("scale", [depth], mean=1.0)
        offset = _biases("offset", [depth])
        mean, variance = tf.nn.moments(input, axes=[1, 2, 3], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def safe_log(x, eps=1e-12):
    return tf.log(x + eps)







