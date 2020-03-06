# _*_ coding:utf-8 _*_
import tensorflow as tf


def _conv2d(name, input, reuse, ngf, pre_layer_ngf, kernel_size=3, strides=1, activation="relu", norm=True):
    """
    Args:
        name: string, id
        input: tensor, [batch_size, width, height, channels]
        reuse: bool, weather to reuse the params
        norm: bool, weather to normalization

    Returns:
        output: tensor, [batch_size, width, height, ngf]
    """
    with tf.variable_scope(name, reuse=reuse):
        if pre_layer_ngf != None:
            conv = tf.layers.conv2d(inputs=input, filters=ngf, kernel_size=kernel_size, strides=strides,
                                    padding="SAME",
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(
                                        mean=1.0 / float(kernel_size * kernel_size * pre_layer_ngf),
                                        stddev=0.000001, dtype=tf.float32),
                                    bias_initializer=tf.constant_initializer(0.0), name=name)
        else:
            conv = tf.layers.conv2d(inputs=input, filters=ngf, kernel_size=kernel_size, strides=strides,
                                    padding="SAME",
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=0.02, dtype=tf.float32),
                                    bias_initializer=tf.constant_initializer(0.0), name=name)

        if norm != False:
            output = _norm(conv)
        else:
            output = conv
        if activation == None:
            pass
        elif activation == "sigmoid":
            output = tf.nn.sigmoid(output)
        else:
            output = _relu(output)
    return output


def _resize_conv(name, input, reuse, shape, ngf, pre_layer_ngf, kernel_size=3, strides=1):
    """
    Args:
        name: string, id
        input: tensor, [batch_size, width, height, channels]
        reuse: bool, weather to reuse the params
    Returns:
        output: tensor, [batch_size, width, height, ngf]
    """
    with tf.variable_scope(name, reuse=reuse):
        resize = _resize(input, reuse=reuse, name='resize', output_size=shape)
        deconv_r = tf.layers.conv2d(inputs=resize, filters=ngf, kernel_size=kernel_size, strides=strides,
                                    padding="SAME",
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(
                                        mean=1.0 / float(kernel_size * kernel_size * pre_layer_ngf),
                                        stddev=0.000001, dtype=tf.float32),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='deconv_r')
        output = _norm(deconv_r)
        return output


def _deconv(name, input, reuse, ngf, pre_layer_ngf, kernel_size=3, strides=2):
    """
    Args:
        name: string, id
        input: tensor, [batch_size, width, height, channels]
        reuse: bool, weather to reuse the params
    Returns:
        output: tensor, [batch_size, width, height, ngf]
    """
    with tf.variable_scope(name, reuse=reuse):
        deconv_t = tf.layers.conv2d_transpose(inputs=input, filters=ngf, kernel_size=kernel_size, strides=strides,
                                              padding="SAME",
                                              activation=None,
                                              kernel_initializer=tf.random_normal_initializer(
                                                  mean=1.0 / float(kernel_size * kernel_size * pre_layer_ngf),
                                                  stddev=0.000001, dtype=tf.float32),
                                              bias_initializer=tf.constant_initializer(0.0),
                                              name='deconv_t')
        output = _norm(deconv_t)
    return output


def _add_conv(name, input1, input2, reuse, ngf, pre_layer_ngf, kernel_size=3, strides=1):
    """
    Args:
        name: string, id
        input1: tensor, [batch_size, width, height, channels]
        input2: tensor, [batch_size, width, height, channels]
        reuse: bool, weather to reuse the params
    Returns:
        output: tensor, [batch_size, width, height, ngf]
    """
    with tf.variable_scope(name, reuse=reuse):
        add = _relu(tf.add(input1 * 0.75, input2 * 0.25))
        add_conv = tf.layers.conv2d(inputs=add, filters=ngf, kernel_size=kernel_size, strides=strides,
                                    padding="SAME",
                                    activation=None,
                                    kernel_initializer=tf.random_normal_initializer(
                                        mean=1.0 / float(kernel_size * kernel_size * pre_layer_ngf),
                                        stddev=0.000001, dtype=tf.float32),
                                    bias_initializer=tf.constant_initializer(0.0),
                                    name='add_conv')
        norm = _norm(add_conv)
        output = _relu(norm)
    return output


def _concat_conv(name, input1, input2, shape, reuse, ngf, pre_layer_ngf, kernel_size=3, strides=1):
    """
    Args:
        name: string, id
        input1: tensor, [batch_size, width, height, channels]
        input2: tensor, [batch_size, width, height, channels]
        reuse: bool, weather to reuse the params
    Returns:
        output: tensor, [batch_size, width, height, ngf]
    """
    with tf.variable_scope(name, reuse=reuse):
        concat = tf.reshape(tf.concat([input1, input2], axis=-1), [shape[0], shape[1], shape[2], 2 * shape[3]])
        conv = tf.layers.conv2d(inputs=concat, filters=ngf, kernel_size=kernel_size, strides=strides,
                                padding="SAME",
                                activation=None,
                                kernel_initializer=tf.random_normal_initializer(
                                    mean=1.0 / float(kernel_size * kernel_size * pre_layer_ngf),
                                    stddev=0.000001, dtype=tf.float32),
                                name='concat_conv')
        norm = _norm(conv)
        output = _relu(norm)
    return output


def _relu(x, alpha=0.2, max_value=100.0):
    '''
    leaky relu
    alpha: slope of negative section.
    '''
    x = tf.maximum(alpha * x, x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))
    return x


def _resize(input, reuse=False, name=None, output_size=None):
    """
    Args:
      input: 4D tensor
      reuse: boolean
      name: string
      output_size: integer list, output size
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
    var = tf.get_variable(name, shape,
                          initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32))
    return var


def _biases(name, shape, constant=0.0):
    """ Helper to create an initialized Bias with constant
    """
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))


def _norm(input):
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
