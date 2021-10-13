import tensorflow as tf


def normalize_2d(x, norm='BN'):
    if norm == 'BN':
        return tf.contrib.layers.batch_norm(x)
    elif norm == 'GN':
        return tf.contrib.layers.group_norm(x, groups=4, reduction_axes=[-3, -2])
    else:
        raise NameError('Undefined normalize method')


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


def conv2d_block(x, channels, norm='BN', activ_func='ReLu', strides=1, kernel_size=3, **kwargs):
    x = tf.layers.conv2d(x, filters=channels, strides=strides, kernel_size=kernel_size, padding='SAME')
    x = normalize_2d(x, norm)
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


def res_block_2d(x, channels=None, norm='BN', activ_func='ReLu', strides=1, kernel_size=3, **kwargs):
    if not channels:
        channels = x.get_shape()[-1]
    mid_channels = channels // 2
    res = conv2d_block(x=x, channels=mid_channels, norm=norm, activ_func=activ_func, strides=strides, kernel_size=1)
    res = conv2d_block(x=res, channels=mid_channels, norm=norm, activ_func=activ_func, strides=1, kernel_size=kernel_size)
    res = conv2d_block(x=res, channels=mid_channels, norm=norm, activ_func=activ_func, strides=1, kernel_size=kernel_size)
    res = conv2d_block(x=res, channels= channels, norm=norm, activ_func=activ_func, strides=1, kernel_size=1)

    short_cut = conv2d_block(x=x, channels=channels, norm=norm, activ_func=activ_func, strides=strides, kernel_size=kernel_size)
    res = res + short_cut
    return res


def group_conv_2d(x, groups=16, channels=128, kernel_size=3, norm='BN'):
    sub_channel = channels / groups
    x_group_list = tf.split(x, num_or_size_splits=groups, axis=-1)
    output_list = []
    for sub_x in x_group_list:
        output_list.append(
            tf.layers.conv2d(sub_x, filters=sub_channel, kernel_size=kernel_size, stride=1, padding='SAME')
        )
    res = tf.concat(output_list, axis=-1)
    res = normalize_2d(res, norm)
    return res


def resX_block_2d(x, channels=None, norm='BN', activ_func='ReLu', strides=1, kernel_size=3):
    if not channels:
        channels = int(x.get_shape()[-1])

    mid_channels = channels // 2
    group = min(mid_channels//2, 16)

    if mid_channels < 4:
        norm = 'BN'

    res = conv2d_block(x=x, channels=mid_channels, norm=norm, activ_func=activ_func, strides=strides, kernel_size=1)
    res = group_conv_2d(x=res, groups=group, channels=mid_channels, kernel_size=kernel_size, norm='BN')
    res = conv2d_block(x=res, channels=channels, norm=norm, activ_func=activ_func, strides=1, kernel_size=1)

    short_cut = conv2d_block(x=x, channels=channels, norm=norm, activ_func=activ_func, strides=strides, kernel_size=kernel_size)
    res = res + short_cut
    return res

