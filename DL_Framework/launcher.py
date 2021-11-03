import warnings
import shutil
from dataTrans import *
from model.model import *
from utils.visual import *
from utils.Initialization import *
import os
import logging
import math
import sys

warnings.filterwarnings('ignore')


class Launcher():
    def __init__(self, params={}):
        self.params = params
        self.train_tfrecord = self.params['train_tfrecord_dir']
        self.val_tfrecord = self.params['val_tfrecord_dir']
        self.train_num = get_tfrecord_sample(self.train_tfrecord)
        self.val_num = get_tfrecord_sample(self.val_tfrecord)
        self.training_iters = math.ceil(self.params['trainset_num'] // self.params['batch_size'])
        self.val_iters = math.ceil(self.params['valset_num'] // self.params['batch_size'])
        self.task = self.params['task']
        self.dimension = self.params['dimension']
        self.is_aug = self.params['is_aug']

    def get_optimizer(self, global_step):
        optimizer_name = self.params.pop('optimizer', 'adam')
        assert optimizer_name in ['adam', 'Momentum', 'RMSProp'], 'Unknown Optimizer'
        if optimizer_name == 'adam':
            learning_rate = self.params.pop('learning_rate', 0.001)
            self.learning_rate_node = tf.Variable(learning_rate, name='learning_rate')
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_node)
        elif optimizer_name == 'Momentum':
            learning_rate = self.params.pop('learning_rate', 0.2)
            decay_rate = self.params.pop('learning_rate_decay', 0.98)
            momentum = self.params.pop('momentum', 0.2)
            self.learning_rate_node = tf.compat.v1.train.experimental_decay(learning_rate=learning_rate,
                                                                            global_step=global_step,
                                                                            decay_steps=self.training_iters,
                                                                            decay_rate=decay_rate,
                                                                            staircase=True
                                                                            )
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum)
        else:
            learning_rate = self.params.pop('learning_rate', 0.001)
            decay_rate = self.params.pop('learning_rate_decay', 0.9)
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.learning_rate_node,
                                                            decay_rate=decay_rate)
        optimizer = optimizer.minimize(self.net.cost, global_step=global_step)
        return optimizer

    def initialize(self):
        global_step = tf.Variable(0, name='global_step')
        tf.compat.v1.summary.scalar('loss', self.net.cost)

        self.optimizer = self.get_optimizer(global_step)
        tf.compat.v1.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.compat.v1.summary.merge_all()
        init = tf.compat.v1.global_variables_initializer()

        self.prediction_directory = self.params.pop('prediction_directory', './tmp_res')
        abs_prediction_path = os.path.abspath(self.prediction_directory)

        self.saver_directory = self.params.pop('saver_directory', './tmp_model')
        abs_saver_path = os.path.abspath(self.saver_directory)

        self.restore = self.params.pop('restore')
        if not self.restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)

            logging.info("Removing '{:}'".format(abs_saver_path))
            shutil.rmtree(abs_saver_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(abs_saver_path):
            logging.info("Allocating '{:}'".format(abs_saver_path))
            os.makedirs(abs_saver_path)

        return init

    def train(self):
        img_size = self.params.pop('img_size')
        label_size = self.params.pop('label_size')
        batch_size = self.params.pop('batch_size', 1)
        epochs = self.params.pop('n_epoch', 50)
        modeltoload = self.params.pop('modeltoload', '')
        n_class = self.params.pop('n_class')
        img_channels = self.params.pop('channels')
        net_name = self.params.pop('net_name')
        basic_layers_name = self.params.pop('basic_layers_name', 'conv')
        weighted_loss = self.params['weighted_loss']
        config = tf.compat.v1.ConfigProto()

        train_tfrecord_path = tf.compat.v1.placeholder(tf.string)
        val_tfrecord_path = tf.compat.v1.placeholder(tf.string)

        train_iterator = make_batch_iterator(train_tfrecord_path, img_size, shuffle=True, batch_size=batch_size, num_class=n_class, aug=True)
        train_img_tf, train_label_tf = train_iterator.get_next()

        val_iterator = make_batch_iterator(val_tfrecord_path, img_size, shuffle=False, batch_size=batch_size, num_class=n_class, aug=False)
        val_img_tf, val_label_tf = val_iterator.get_next()

        if self.task == 'Segmentation':
            loss_function = self.params.pop('loss_function', 'dice_loss')
            score_index = self.params.pop('score_index', 'Dice')

            self.net = Segmentation_Model(train_img_tf, train_label_tf, n_class, img_channels, net_name, basic_layers_name, loss_function, weighted_loss, score_index,
                                          True)
            self.vnet = Segmentation_Model(val_img_tf, val_label_tf, n_class, img_channels, net_name, basic_layers_name, loss_function, weighted_loss, score_index,
                                           False)
        init = self.initialize()
        save_path = os.path.join(self.saver_directory, "model.ckpt")
        if epochs == 0:
            return save_path
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(init)
            if self.restore and modeltoload:
                self.net.restore(sess, modeltoload)
            summary_writer = tf.compat.v1.summary.FileWriter(self.saver_directory, graph=sess.graph)
            logging.info("Start Optimization")
            for epoch in range(self.epochs):
                sess.run(train_iterator.initializer, feed_dict={train_tfrecord_path: self.train_tfrecord})
                total_train_loss, train_iters_num = 0, 0
                while True:
                    try:
                        try:
                            train_iters_num += 1
                            _, cur_train_loss, lr = sess.run([self.optimizer, self.net.cost, self.learning_rate_node])
                            total_train_loss += cur_train_loss
                            vis_train_log(train_iters_num, epochs, epoch, batch_size, self.train_num, 'dice_loss',
                                          total_train_loss)
                        except tf.errors.DataLossError as e:
                            logging.info('******DataLoss******')
                    except tf.errors.OutOfRangeError as e:
                        sys.stdout.write('\n')
                        break
                avg_train_loss = total_train_loss / self.training_iters
                save_path = self.net.save(sess, os.path.join(self.saver_directory,
                                                             "epoch_%03d_%.04f.model.ckpt" % (epoch, avg_train_loss)))
                if epoch % 5 == 0:
                    sess.run(val_iterator.initializer, feed_dict={val_tfrecord_path: self.val_tfrecord})
                    total_val_score, val_iters_num = 0, 0
                    while True:
                        try:
                            try:
                                val_iters_num += 1
                                cur_val_score = sess.run(self.vnet.score)
                                total_val_score += cur_val_score
                                vis_val_log(val_iters_num, epoch, batch_size, self.val_num, 'Dice',
                                            total_val_score)
                            except tf.errors.DataLossError as e:
                                logging.info('******DataLoss******')
                        except tf.errors.OutOfRangeError as e:
                            sys.stdout.write('\n')
            print("Optimization Finished!")
            return save_path
