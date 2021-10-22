from model.basic_layers import *

def unet_3d(x, channels=32, n_class=2, is_training=True, basic_layers_name='conv', reuse=tf.AUTO_REUSE, drop_rate=0.2):
    if basic_layers_name == 'conv':
        basic_layers = conv3d_block
    with tf.variable_scope('unet_3d', reuse=reuse):
        D0 = tf.identity(x, name='network_input')
        with tf.variable_scope('D1', reuse=reuse):
            D1 = basic_layers(D0, channels)
            D1 = basic_layers(D1, channels)
            D1 = basic_layers(D1, channels)
        with tf.variable_scope('D2', reuse=reuse):
            D2 = basic_layers(D1, channels * 2, strides=[2, 2, 2])
            D2 = basic_layers(D2, channels * 2)
        with tf.variable_scope('D3', reuse=reuse):
            D3 = basic_layers(D2, channels * 4, strides=[2, 2, 2])
            D3 = basic_layers(D3, channels * 4)
        with tf.variable_scope('middle', reuse=reuse):
            MD = basic_layers(D3, channels * 8, strides=[2, 2, 2])
            MD = basic_layers(MD, channels * 8)
        with tf.variable_scope('U3', reuse=reuse):
            U3 = upsampling_3d(MD, 'deconv', [2, 2, 2], channels * 4)
            U3 = tf.concat([U3, D3], axis=-1)
            U3 = basic_layers(U3, channels * 4)
            U3 = drop_out(U3, drop_rate, is_training)
            U3 = basic_layers(U3, channels * 4)
        with tf.variable_scope('U2', reuse=reuse):
            U2 = upsampling_3d(U3, 'deconv', [2, 2, 2], channels * 2)
            U2 = tf.concat([U2, D2], axis=-1)
            U2 = basic_layers(U2, channels * 2)
            U2 = drop_out(U2, drop_rate, is_training)
            U2 = basic_layers(U2, channels * 2)
        with tf.variable_scope('U1', reuse=reuse):
            U1 = upsampling_3d(U2, 'deconv', [2, 2, 2], channels * 2)
            U1 = tf.concat([U1, D1], axis=-1)
            U1 = conv2d_block(U1, channels * 2)
            U1 = drop_out(U1, drop_rate, is_training)
            U1 = basic_layers(U1, channels * 2)
        with tf.variable_scope('OP', reuse=reuse):
            OP = basic_layers(U1, n_class)
            OP = basic_layers(OP, n_class)
            OP = tf.nn.softmax(OP, axis=-1)
        OP = tf.identity(OP, name='network_output')
        return OP


def unet_2d(x, channels=32, n_class=2, is_training=True, basic_layers_name='conv', reuse=tf.AUTO_REUSE, drop_rate=0.2):
    if basic_layers_name == 'conv':
        basic_layers= conv2d_block
    elif basic_layers_name == 'res_block':
        basic_layers = res_block_2d
    elif basic_layers_name == 'resX_block':
        basic_layers = resX_block_2d
    else:
        raise NameError('Undefined basic_layers_name!')

    with tf.variable_scope('unet_2d', reuse=reuse):
        D0 = tf.identity(x, name='network_input')
        with tf.variable_scope('D1', reuse=reuse):
            D1 = basic_layers(D0, channels)
            D1 = basic_layers(D1, channels)
            D1 = basic_layers(D1, channels)
        with tf.variable_scope('D2', reuse=reuse):
            D2 = basic_layers(D1, channels * 2, strides=2)
            D2 = basic_layers(D2, channels * 2)
        with tf.variable_scope('D3', reuse=reuse):
            D3 = basic_layers(D2, channels * 4, strides=2)
            D3 = basic_layers(D3, channels * 4)
        with tf.variable_scope('middle', reuse=reuse):
            MD = basic_layers(D3, channels * 8, strides=2)
            MD = basic_layers(MD, channels * 8)
        with tf.variable_scope('U3', reuse=reuse):
            U3 = upsampling_2d(MD, 'deconv', 2, channels * 4)
            U3 = tf.concat([U3, D3], axis=-1)
            U3 = basic_layers(U3, channels * 4)
            U3 = drop_out(U3, drop_rate, is_training)
            U3 = basic_layers(U3, channels * 4)
        with tf.variable_scope('U2', reuse=reuse):
            U2 = upsampling_2d(U3, 'deconv', 2, channels * 2)
            U2 = tf.concat([U2, D2], axis=-1)
            U2 = basic_layers(U2, channels * 2)
            U2 = drop_out(U2, drop_rate, is_training)
            U2 = basic_layers(U2, channels * 2)
        with tf.variable_scope('U1', reuse=reuse):
            U1 = upsampling_2d(U2, 'deconv', 2, channels * 2)
            U1 = tf.concat([U1, D1], axis=-1)
            U1 = conv2d_block(U1, channels * 2)
            U1 = drop_out(U1, drop_rate, is_training)
            U1 = basic_layers(U1, channels * 2)
        with tf.variable_scope('OP', reuse=reuse):
            OP = basic_layers(U1, n_class)
            OP = basic_layers(OP, n_class)
            OP = tf.nn.softmax(OP, axis=-1)
        OP = tf.identity(OP, name='network_output')
        return OP
