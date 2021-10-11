from model.Seg_model import *
from model.loss import *


class Segmentation_Model:
    def __init__(self, img_tf, label_tf, num_class, channels, net_name, loss_function, is_training):
        self.x = img_tf
        self.y = tf.cast(label_tf, tf.float32)
        self.num_class = num_class
        self.channels = channels
        self.net_name = net_name
        self.loss_function = loss_function
        self.is_training = is_training
        self.cost = self.get_loss()

    def inference(self):
        self.pred = unet_2d(self.x, 16, 5, self.is_training)
        return self.pred

    def get_loss(self):
        self.inference()
        if self.loss_function == 'dice_loss':
            loss = dice_loss(self.pred, self.y)
        return loss

    def save(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, model_path)
        print("Model restored from file: %" % model_path)