import tensorflow as tf


def activate(x, activ_func):
    if activ_func == 'ReLu':
        return tf.nn.relu(x)
    elif activ_func == 'Leaky_ReLu':
        return tf.nn.leaky_relu(x)
    else:
        raise NameError('Undefined Activation Function')


def drop_out(x, dropout_rate, is_training=True, scope='dropout'):
    with tf.variable_scope(scope):
        x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)
        return x


def conv2d_block(x, channels, norm='BN', activ_func='ReLu', stride=1, kernel_size=3):
    x = tf.layers.conv2d(x, filters=channels, strides=stride, kernel_size=kernel_size, padding='SAME')
    if norm == 'GN':
        x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=[-3, -2])
    elif norm == 'BN':
        x = tf.contrib.layers.batch_norm(x)
    else:
        raise NameError('Undefined normalize method')
    x = activate(x, activ_func=activ_func)
    return x


def upsampling_2d(x, up_sampling_method, scale_factor, channels, norm='BN', activ_func='ReLu'):
    if up_sampling_method == 'deconv':
        x = tf.layers.conv2d_transpose(x, filters=channels, strides=scale_factor, kernel_size=scale_factor * 2,
                                       padding='SAME')
    elif up_sampling_method == 'interpolation':
        x = tf.image.resize_images(x, [x.shape[1] * scale_factor, x.shape[2] * scale_factor])
        x = tf.layers.conv2d(x, filters=channels, stride=1, kernel_size=3, padding='SAME')
    else:
        raise NameError('Undefined upsampling method')
    if norm == 'GN':
        x = tf.contrib.layers.group_norm(x, groups=4, reduction_axes=[-3, -2])
    elif norm == 'BN':
        x = tf.contrib.layers.batch_norm(x)
    else:
        raise NameError('Undefined normalize method')
    x = activate(x, activ_func=activ_func)
    return x
