import numpy as np
import sys


def vis_train_log(iters, total_epoch, cur_epoch, batch_size, train_num, loss_function, total_loss):
    percent = iters * batch_size / train_num * 100
    num = np.int(20 * percent / 100)
    avg_loss = total_loss / iters
    sys.stdout.write(
        '\r Epoch:[{0}'.format(cur_epoch + 1) + '/{0}]'.format(total_epoch) + '{0}>'.format("=" * num) + "{0}>".format(
            " " * (20 - num)) + '||' + str(
            percent)[:4] + '%' + ' [{0}:'.format(iters * batch_size) + '{0}]'.format(
            train_num) + ' ' + loss_function + ':' + ' {:.4f}'.format(total_loss) + ' Avg_loss: {:.4f}'.format(
            total_loss / (avg_loss)))
    sys.stdout.flush()


def vis_val_log(iters, cur_epoch, batch_size, val_num, val_score_name, total_score):
    percent = iters * batch_size / val_num * 100
    num = np.int(20 * percent / 100)
    avg_score = total_score / iters
    sys.stdout.write(
        '\r Validation Starting:  Epoch[{0}]'.format(cur_epoch + 1) + '{0}>'.format("=" * num) + "{0}>".format(
            " " * (20 - num)) + '||' + str(
            percent)[:4] + '%' + ' [{0}:'.format(iters * batch_size) + '{0}]'.format(
            val_num) + ' ' + val_score_name + ':' + ' {:.4f}'.format(total_score) + ' Avg_score: {:.4f}'.format(
            total_score / (avg_score)))
    sys.stdout.flush()
