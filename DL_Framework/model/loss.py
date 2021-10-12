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
