import numpy as np
import tensorflow as tf


def parse_function(serialized_example, image_size):
    image_length = np.prod(np.array(image_size))
    features = {
        'img': tf.FixedLenFeature([image_length], tf.float32),
        'label': tf.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.io.parse_single_example(serialized_example, features)
    img = parsed_example['img']
    label = tf.decode_raw(parsed_example['label'], tf.uint8)
    return img, label


def make_batch_iterator(tfrecord_path, image_size, shuffle=True, batch_size=1):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    if shuffle:
        dataset = dataset.shuffle(10)
    dataset = dataset.map(lambda x: parse_function(x, image_size), num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(-1)
    return dataset.make_initializable_iterator()
