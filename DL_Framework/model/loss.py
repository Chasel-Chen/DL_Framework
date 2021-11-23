import tensorflow as tf


def dice_loss(y_pred, gt):
    g1 = tf.reshape(y_pred, [-1, y_pred.shape[-1]])[..., 1:]
    p1 = tf.reshape(gt, [-1, y_pred.shape[-1]])[..., 1:]
    intersection = tf.reduce_sum(g1 * p1)
    union = tf.reduce_sum(g1) + tf.reduce_sum(p1)
    dice = (2 * intersection + 1.) / (union + 1.)
    return 1 - dice


def dice_score(y_pred, gt):
    g1 = tf.reshape(y_pred, [-1, y_pred.shape[-1]])[..., 1:]
    p1 = tf.reshape(gt, [-1, y_pred.shape[-1]])[..., 1:]
    intersection = tf.reduce_sum(g1 * p1)
    union = tf.reduce_sum(g1) + tf.reduce_sum(p1)
    dice = (2 * intersection + 1.) / (union + 1.)
    return dice


def multi_dice_score(y_pred, gt):
    g1 = tf.reshape(y_pred, [-1, y_pred.shape[-1]])[..., 1:]
    p1 = tf.reshape(gt, [-1, y_pred.shape[-1]])[..., 1:]
    intersection = tf.reduce_sum(g1 * p1, axis=(0, 1, 2))
    union = tf.reduce_sum(g1, axis=(0, 1, 2)) + tf.reduce_sum(p1, axis=(0, 1, 2))
    dice = (2 * intersection + 1.) / (union + 1.)
    return dice


def explog_loss(y_pred, gt, num_class, weight=None):
    if not weight:
        weight = [1.] * num_class
    if len(weight) != num_class:
        raise ValueError('The length of weight should be same as num_class')
    smooth = 1
    intersection = tf.reduce_mean(y_pred * gt, axis=(0, 1, 2))
    sum_of_area = tf.reduce_mean(y_pred + gt, axis=(0, 1, 2))
    dice = (2 * intersection + smooth) / (sum_of_area + smooth)
    dice_loss = tf.reduce_mean((-tf.log(dice[1:])) ** 0.3)
    weights = tf.constant(weight, shape=[1, 1, 1, num_class])
    eps = 1e-6
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    res = tf.reduce_sum(gt*(-tf.log(y_pred))**0.3*weights, axis=-1)
    res = tf.reduce_mean(res)
    return 0.8 * dice_loss + 0.2 * res
