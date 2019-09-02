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


def norm(input):
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
