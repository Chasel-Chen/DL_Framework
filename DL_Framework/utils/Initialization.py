import warnings
import configparser
import tensorflow as tf

warnings.filterwarnings('ignore')


def process_config(config_file):
    params = {}
    config = configparser.ConfigParser()
    config.read(config_file)
    for section in config.sections():
        if section == 'Task':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'DataSet':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params


def get_tfrecord_sample(file_name):
    sample_nums = 0
    for _ in tf.python_io.tf_record_iterator(file_name):
        sample_nums += 1
    return sample_nums