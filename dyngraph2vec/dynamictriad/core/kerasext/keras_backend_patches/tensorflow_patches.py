import tensorflow as tf


def gather_nd(param, indices):
    return tf.gather_nd(param, indices)


def slice(input, begin, size):
    return tf.slice(input, begin, size)


def log_softmax(logits, axis=-1):
    return tf.nn.log_softmax(logits, axis)
