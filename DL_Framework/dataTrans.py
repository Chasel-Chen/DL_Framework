import tensorflow as tf
import math
import elasticdeform.tf as etf


def parse_function(tfr):
    example = tf.parse_single_example(tfr,
                                      features={
                                          "img": tf.FixedLenFeature([], tf.string),
                                          "mask": tf.FixedLenFeature([], tf.string)})
    img = tf.decode_raw(example["img"], tf.float32)
    mask = tf.decode_raw(example["mask"], tf.uint8)
    return img, mask


def preprocess(img, mask, image_size):
    img = tf.reshape(img, image_size)
    mask = tf.reshape(mask, image_size)
    return img, mask


def data_norm(img, methods='z_score'):
    if methods == 'z_score':
        axes = list(range(len(img.shape)))
        mean, var = tf.nn.moments(img, axes)
        std = tf.sqrt(var)
        img = (img - mean) / std
    elif methods == "min_max":
        max_v = tf.reduce_max(img)
        min_v = tf.reduce_min(img)
        img = (img - min_v) / (max_v - min_v)
    elif methods == 'clip':
        img = tf.clip_by_value(img, 500, 1400)
        img = (img - 500.) / 900.
    else:
        raise NameError('Undefined data_norm_method!')
    return img


def augment(img, mask, aug=True, rr=(-30, 30), ed=(3, 15), tr=(-20, 20), flip=0.5, num_class=2):
    if not aug:
        img = data_norm(img, 'z_score')
        img = tf.expand_dims(data_norm(img), -1)
        mask = tf.one_hot(tf.cast(mask, tf.uint8), num_class, axis=-1, dtype=tf.float32)
        return img, mask

    # Elastic Deform
    s = img.get_shape().as_list()
    m = tf.random_normal((2, ed[0], ed[0]), stddev=ed[1])
    img = etf.deform_grid(img, m, order=3, axis=(0, 1))
    mask = etf.deform_grid(mask, m, order=0, axis=(0, 1))
    img = tf.reshape(img, s)
    mask = tf.reshape(mask, s)

    # Rotate
    r = tf.random_uniform((), rr[0], rr[1])
    r = r * math.pi / 180
    img = tf.contrib.image.rotate(img, r, 'BILINEAR')
    mask = tf.contrib.image.rotate(mask, r, 'NEAREST')

    # Translate
    tx = tf.random_uniform((), tr[0], tr[1])
    ty = tf.random_uniform((), tr[0], tr[1])
    img = tf.contrib.image.translate(img, (tx, ty), 'BILINEAR')
    mask = tf.contrib.image.translate(mask, (tx, ty), 'NEAREST')

    img = data_norm(img, 'z_score')
    img = tf.expand_dims(img, -1)
    mask = tf.one_hot(tf.cast(mask, tf.uint8), num_class, axis=-1, dtype=tf.float32)

    # Flip horizontal
    p = tf.random_uniform([], 0, 1)
    img = tf.cond(p > flip, lambda: tf.image.flip_left_right(img), lambda: img)
    mask = tf.cond(p > flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    return img, mask


def make_batch_iterator(tfr, image_size, shuffle=True, batch_size=1, num_class=2, if_aug=True):
    dataset = tf.data.TFRecordDataset(tfr)
    if shuffle:
        dataset = dataset.shuffle(10)
    dataset = dataset.map(parse_function)
    dataset = dataset.map(lambda img, mask: preprocess(img, mask, image_size))
    dataset = dataset.map(lambda img, mask: augment(img, mask, if_aug, num_class=num_class))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(-1)
    return dataset.make_initializable_iterator()
