from model.Seg_model import *
from model.loss import *


class Segmentation_Model:
    def __init__(self, img_tf, label_tf, num_class, channels, net_name, basic_layers_name='conv' ,loss_function='dice_loss', score_index='Dice', is_training=True):
        self.x = img_tf
        self.y = tf.cast(label_tf, tf.float32)
        self.num_class = num_class
        self.channels = channels
        self.net_name = net_name
        self.basic_layers_name = basic_layers_name
        self.loss_function = loss_function
        self.score_index = score_index
        self.is_training = is_training
        self.cost = self.get_loss()
        self.score = self.get_score()

    def inference(self):
        if self.net_name == "unet_2d":
            self.pred = unet_2d(self.x, 16, self.num_class, self.is_training, self.basic_layers_name)
        return self.pred

    def get_loss(self):
        self.inference()
        if self.loss_function == 'dice_loss':
            loss = dice_loss(self.pred, self.y)
        elif self.loss_function == 'explog_loss':
            loss = explog_loss(self.pred, self.y)
        return loss

    def get_score(self):
        self.inference()
        if self.score_index == 'Dice':
            score = dice_score(self.pred, self.y)
        return score

    def save(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, model_path)
        print("Model restored from file: %" % model_path)