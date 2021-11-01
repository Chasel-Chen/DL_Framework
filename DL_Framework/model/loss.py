import tensorflow as tf


def dice_loss(y_pred, gt):
    g1 = tf.reshape(y_pred, [-1, y_pred.shape[-1]])[..., 1:]
    p1 = tf.reshape(gt, [-1, y_pred.shape[-1]])[..., 1:]
    intersection = tf.reduce_sum(g1 * p1)
    union = tf.reduce_sum(g1) + tf.reduce_sum(p1)
    dice_score = (2 * intersection + 1.) / (union + 1.)
    return 1 - dice_score


def dice_score(y_pred, gt):
    g1 = tf.reshape(y_pred, [-1, y_pred.shape[-1]])[..., 1:]
    p1 = tf.reshape(gt, [-1, y_pred.shape[-1]])[..., 1:]
    intersection = tf.reduce_sum(g1 * p1)
    union = tf.reduce_sum(g1) + tf.reduce_sum(p1)
    dice_score = (2 * intersection + 1.) / (union + 1.)
    return dice_score


def explog_loss(y_pred, gt):
    smooth = 1
    intersection = tf.reduce_mean(y_pred * gt, axis=(0, 1, 2))
    sum_of_area = tf.reduce_mean(y_pred + gt, axis=(0, 1, 2))
    dices = (2 * intersection + smooth) / (sum_of_area + smooth)
    dice_loss = tf.reduce_mean((-tf.log(dices[1:])) ** 0.3)
    weights = tf.constant([1.10177024, 2.38225372], shape = [1, 1, 1, 2])
    eps = 1e-6
    y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
    res = tf.reduce_sum(gt*(-tf.log(y_pred))**0.3*weights, axis=-1)
    res = tf.reduce_mean(res)
    return 0.8 * dice_loss + 0.2 * res