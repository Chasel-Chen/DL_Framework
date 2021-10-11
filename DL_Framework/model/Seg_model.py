from model.basic_layers import *


def unet_2d(x, channels=16, n_class=2, is_training=True, reuse=tf.AUTO_REUSE, drop_rate=0.2):
    with tf.variable_scope('network', reuse=reuse):
        D0 = tf.identity(x, name='network_input')
        with tf.variable_scope('D1', reuse=reuse):
            D1 = conv2d_block(D0, channels)
            D1 = conv2d_block(D1, channels)
        with tf.variable_scope('D2', reuse=reuse):
            D2 = conv2d_block(D1, channels * 2, stride=2)
            D2 = conv2d_block(D2, channels * 2)
        with tf.variable_scope('D3', reuse=reuse):
            D3 = conv2d_block(D2, channels * 4, stride=2)
            D3 = conv2d_block(D3, channels * 4)
        with tf.variable_scope('middle', reuse=reuse):
            MD = conv2d_block(D3, channels * 8, stride=2)
            MD = conv2d_block(MD, channels * 8)
            MD = conv2d_block(MD, channels * 8)
        with tf.variable_scope('U3', reuse=reuse):
            U3 = upsampling_2d(MD, 'deconv', 2, channels * 4)
            U3 = tf.concat([U3, D3], axis=-1)
            U3 = conv2d_block(U3, channels * 4)
            U3 = drop_out(U3, drop_rate, is_training)
            U3 = conv2d_block(U3, channels * 4)
        with tf.variable_scope('U2', reuse=reuse):
            U2 = upsampling_2d(U3, 'deconv', 2, channels * 2)
            U2 = tf.concat([U2, D2], axis=-1)
            U2 = conv2d_block(U2, channels * 2)
            U2 = drop_out(U2, drop_rate, is_training)
            U2 = conv2d_block(U2, channels * 2)
        with tf.variable_scope('U1', reuse=reuse):
            U1 = upsampling_2d(U2, 'deconv', 2, channels * 2)
            U1 = tf.concat([U1, D1], axis=-1)
            U1 = conv2d_block(U1, channels * 2)
            U1 = drop_out(U1, drop_rate, is_training)
            U1 = conv2d_block(U1, channels * 2)
        with tf.variable_scope('OP', reuse=reuse):
            OP = conv2d_block(U1, n_class)
            OP = conv2d_block(OP, n_class)
            OP = tf.nn.softmax(OP, axis=-1)
        OP = tf.identity(OP, name='network_output')
        return OP
