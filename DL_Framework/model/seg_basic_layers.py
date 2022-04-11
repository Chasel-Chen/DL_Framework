from model.basic_layers import *


def bisenet_feature_fusion_model(feat_sp, feat_cp, out_channel):
    fcat = tf.concat([feat_sp, feat_cp], dim=-1)
    feat = conv2d_block(fcat, channels=out_channel, norm='BN', strides=1, kernel_size=1)
    atten = tf.reduce_mean(feat, axis=[1,2])
    atten = tf.layers.conv2d(atten, filters=out_channel, strides=1, kernel_size=1, padding='SAME', use_bias=False)
    atten = tf.nn.sigmoid(normalize_2d(atten))
    feat_atten = feat*atten
    feat_out = feat_atten + feat
    return feat_out


def bisenet_spatial_path(x, middle_channels=64, out_channels=128):
    feat = conv2d_block(x, channels=middle_channels, kernel_size=7, strides=2)
    feat = conv2d_block(feat, channels=middle_channels, kernel_size=3, strides=2)
    feat = conv2d_block(feat, channels=middle_channels, kernel_size=3, strides=2)
    feat = conv2d_block(feat, channels=out_channels, kernel_size=1, strides=1)
    return feat


def bisenet_attention_refinement_module(x, out_channels):
    feat = conv2d_block(x, channels=out_channels, kernel_size=3, strides=1)
    atten = tf.reduce_mean(feat, axis = [1,2])
    atten = tf.layers.conv2d(atten, filters=out_channels, strides=1, kernel_size=1, padding='SAME', use_bias=False)
    atten = tf.nn.sigmoid(normalize_2d(atten))
    out = feat*atten
    return out


def bisenet_context_path(x):
    feat8, feat16, feat32 = resnet18(x)
    avg = tf.reduce_mean(feat32, axis=[1,2])
    avg = conv2d_block(avg, channels=128, kernel_size=1, strides=1)

    feat32_arm = bisenet_attention_refinement_module(feat32, 128)
    feat32_sum = feat32_arm + avg
    feat32_up = tf.depth_to_space(tf.layers.conv2d(feat32_sum, filters=128*4, strides=1, kernel_size=1, padding='SAME'), 2)
    feat32_up = tf.conv2d_block(feat32_up, 128, norm='BN', activ_func='ReLu', strides=1, kernel_size=3)

    feat16_arm = bisenet_attention_refinement_module(feat16, 128)
    feat16_sum = feat16_arm + feat32_up
    feat16_up = tf.depth_to_space(tf.layers.conv2d(feat16_sum, filters=128*4, strides=1, kernel_size=1, padding='SAME'), 2)
    feat16_up = tf.conv2d_block(feat16_up, 128, norm='BN', activ_func='ReLu', strides=1, kernel_size=3)

    return feat32_up, feat16_up
