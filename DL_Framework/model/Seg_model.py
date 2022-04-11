from model.basic_layers import *
from model.seg_basic_layers import *


def unet_2d(x, channels=32, n_class=2, is_training=True, basic_layers_name='conv', reuse=tf.AUTO_REUSE, drop_rate=0.2):
    if basic_layers_name == 'conv':
        basic_layers = conv2d_block
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
        with tf.variable_scope('D2', reuse=reuse):
            D2 = tf.nn.max_pool(D1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            D2 = basic_layers(D2, channels * 2)
            D2 = basic_layers(D2, channels * 2)
        with tf.variable_scope('D3', reuse=reuse):
            D3 = tf.nn.max_pool(D2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            D3 = basic_layers(D3, channels * 4)
            D3 = basic_layers(D3, channels * 4)
        with tf.variable_scope('D4', reuse=reuse):
            D4 = tf.nn.max_pool(D3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            D4 = basic_layers(D4, channels * 8)
            D4 = basic_layers(D4, channels * 8)
        with tf.variable_scope('D5', reuse=reuse):
            D5 = tf.nn.max_pool(D4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            D5 = basic_layers(D5, channels * 16)
            D5 = basic_layers(D5, channels * 16)
        with tf.variable_scope('middle', reuse=reuse):
            MD = tf.nn.max_pool(D5, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            MD = basic_layers(MD, channels * 32)
            MD = basic_layers(MD, channels * 32)
        with tf.variable_scope('U5', reuse=reuse):
            U5 = upsampling_2d(MD, 'deconv', 2, channels * 16)
            U5 = tf.concat([U5, D5], axis=-1)
            U5 = drop_out(U5, drop_rate, is_training)
            U5 = basic_layers(U5, channels * 16)
            U5 = basic_layers(U5, channels * 16)
        with tf.variable_scope('U4', reuse=reuse):
            U4 = upsampling_2d(U5, 'deconv', 2, channels * 8)
            U4 = tf.concat([U4, D4], axis=-1)
            U4 = drop_out(U4, drop_rate, is_training)
            U4 = basic_layers(U4, channels * 8)
            U4 = basic_layers(U4, channels * 8)
        with tf.variable_scope('U3', reuse=reuse):
            U3 = upsampling_2d(MD, 'deconv', 2, channels * 4)
            U3 = tf.concat([U3, D3], axis=-1)
            U3 = drop_out(U3, drop_rate, is_training)
            U3 = basic_layers(U3, channels * 4)
            U3 = basic_layers(U3, channels * 4)
        with tf.variable_scope('U2', reuse=reuse):
            U2 = upsampling_2d(U3, 'deconv', 2, channels * 2)
            U2 = tf.concat([U2, D2], axis=-1)
            U2 = drop_out(U2, drop_rate, is_training)
            U2 = basic_layers(U2, channels * 2)
            U2 = basic_layers(U2, channels * 2)
        with tf.variable_scope('U1', reuse=reuse):
            U1 = upsampling_2d(U2, 'deconv', 2, channels * 2)
            U1 = tf.concat([U1, D1], axis=-1)
            U1 = conv2d_block(U1, channels * 2)
            U1 = drop_out(U1, drop_rate, is_training)
            U1 = basic_layers(U1, channels * 2)
        with tf.variable_scope('OP', reuse=reuse):
            OP = basic_layers(U1, n_class, kernel_size=[1,1])
            OP = tf.nn.softmax(OP, axis=-1)
        OP = tf.identity(OP, name='network_output')
        return OP


def bisenet_2d_v1(x, n_class=2, is_training=True, reuse=tf.AUTO_REUSE):
    s = x.get_shape().as_list()
    feat_cp8, feat_cp16 = bisenet_context_path(x)
    feat_sp = bisenet_spatial_path(x)
    feat_fuse = featurefusionmodule(feat_sp, feat_cp8)
    feat_out = conv_out(feat_fuse)
    if is_training:
        feat_out16 = conv_out16(feat_cp8)
        feat_out32 = conv_out32(feat_cp16)
        return feat_out, feat_out16, feat_out32
    else:
        return feat_out
