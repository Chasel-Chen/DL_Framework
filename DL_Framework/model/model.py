from model.loss import *
from model.basic_layers import *


class Unet2d:
    def __init__(self, img_tf, label_tf, num_class, channels, basic_layers_name='conv', loss_function='dice_loss',
                 weighted_loss=None, score_index='Dice', is_traning=True):
        self.x = img_tf
        self.y = label_tf
        self.weighted_loss = weighted_loss
        self.num_class = num_class
        self.channels = channels
        self.basic_layers_name = basic_layers_name
        self.loss_function = loss_function
        self.score_index = score_index
        self.is_training = is_traning
        self.pred = self.inference(self.x, self.channels, self.num_class, self.is_training, self.basic_layers_name)
        self.cost = self.get_loss()
        self.score = self.get_score()

    def down_stage(self, x, basic_layers, channels, name_scope, reuse):
        with tf.variable_scope(name_scope, reuse=reuse):
            x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            x = basic_layers(x, channels)
            x = basic_layers(x, channels)
        return x

    def up_stage(self, x_l, x_h, basic_layers, channels, name_scope, up_method, reuse):
        with tf.variable_scope(name_scope, reuse=reuse):
            x = upsampling_2d(x_l, up_method, 2, channels)
            x = tf.concat([x, x_h], axis=-1)
            x = basic_layers(x, channels)
            x = basic_layers(x, channels)
            return x

    def inference(self, x, channels=32, n_class=2, is_training=True, basic_layers_name='conv', reuse=tf.AUTO_REUSE):
        if basic_layers_name == 'conv':
            basic_layers = conv2d_block
        elif basic_layers_name == 'res_block':
            basic_layers = res_block_2d
        elif basic_layers_name == 'resX_block':
            basic_layers = resX_block_2d
        else:
            raise NameError('Undefined basic layers name !!')
        with tf.variable_scope('unet_2d', reuse=reuse):
            D0 = tf.identity(x, name='network_input')
            with tf.variable_scope('D1', reuse=reuse):
                D1 = basic_layers(D0, channels)
                D1 = basic_layers(D1, channels)
            D2 = self.down_stage(D1, basic_layers, channels * 2, 'D2', reuse=reuse)
            D3 = self.down_stage(D2, basic_layers, channels * 4, 'D3', reuse=reuse)
            D4 = self.down_stage(D3, basic_layers, channels * 8, 'D4', reuse=reuse)
            D5 = self.down_stage(D4, basic_layers, channels * 16, 'D5', reuse=reuse)
            MD = self.down_stage(D5, basic_layers, channels * 32, 'MD', reuse=reuse)
            U5 = self.up_stage(MD, D5, basic_layers, channels * 16, 'U5', up_method='deconv', reuse=reuse)
            U4 = self.up_stage(U5, D4, basic_layers, channels * 8, 'U4', up_method='deconv', reuse=reuse)
            U3 = self.up_stage(U4, D3, basic_layers, channels * 4, 'U3', up_method='deconv', reuse=reuse)
            U2 = self.up_stage(U3, D2, basic_layers, channels * 2, 'U2', up_method='deconv', reuse=reuse)
            U1 = self.up_stage(U2, D1, basic_layers, channels, 'U1', up_method='deconv', reuse=reuse)
            with tf.variable_scope('Output', reuse=reuse):
                OP = basic_layers(U1, n_class, kernel_size=[1, 1])
                OP = tf.nn.softmax(OP, axis=-1)
            OP = tf.identity(OP, name='Network_Output')
            return OP

    def get_loss(self):
        if self.loss_function == 'dice_loss':
            loss = dice_loss(self.pred, self.y)
        elif self.loss_function == 'explog_loss':
            loss = explog_loss(self.pred.self.y, self.num_class, self.weighted_loss)
        else:
            raise NameError('Undefined loss function')
        return loss

    def get_score(self):
        if self.score_index == 'Dice':
            score = dice_score(self.pred, self.y)
        elif self.score_index == 'Multi_Dice':
            score = multi_dice_score(self.pred, self.y)
        else:
            raise NameError('Undefined score index')
        return score

    def save(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, model_path)
        print('Model restored from file, %s' % model_path)
