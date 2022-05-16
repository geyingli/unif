""" Version control of dependencies. """

import tensorflow as tf


if tf.__version__.startswith("2"):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
