import re
import numpy as np

from ..tools import tf


def get_grad_and_param(variables, grads, param_name):
    for (grad, param) in zip(grads, variables):
        if param_name in param.name:
            return (grad, param)
    return None, None


def get_param(variables, param_name):
    for param in variables:
        if param_name in param.name:
            return param
    return None


def get_param_name(param):
    res = re.match("^(.*):\\d+$", param.name)
    if res is not None:
        param_name = res.group(1)
    return param_name


def count_params(global_variables, trainable_variables):
    def get_params(variable):
        _tuple = tuple(map(int, variable.shape))
        if not _tuple:
            return 0
        return np.prod(_tuple)
    n_global = 0
    for variable in global_variables:
        n_global += get_params(variable)
    n_trainable = 0
    for variable in trainable_variables:
        n_trainable += get_params(variable)
    tf.logging.info("Build graph with %s parameters "
                    "(among which %s are trainable)"
                    % (format(int(n_global), ","),
                       format(int(n_trainable), ",")))


def scale_grad(grad, scalar):
    if grad is None:
        return None

    if grad.__str__().startswith("IndexedSlices"):
        return tf.IndexedSlices(
            values=grad.values * scalar,
            indices=grad.indices,
            dense_shape=grad.dense_shape)
    else:
        return grad * scalar


def add_n_grads(split_grads):
    split_grads = [grad for grad in split_grads if grad is not None]
    if len(split_grads) == 1:
        return split_grads[0]

    # Dealing with IndexedSlices for large-dimensional embedding
    # matrix. The gradient of an embedding matrix is not a tensor,
    # but a tuple-like object named `IndexedSlices`, for this one,
    # we need to take special processings.
    if split_grads[0].__str__().startswith("IndexedSlices"):

        values = tf.concat([grad.values for grad in split_grads], axis=0)
        indices = tf.concat([grad.indices for grad in split_grads], axis=0)
        dense_shape = split_grads[0].dense_shape

        return tf.IndexedSlices(
            values=values,
            indices=indices,
            dense_shape=dense_shape)

    return tf.add_n(split_grads)


def average_n_grads(split_grads):
    split_grads = [grad for grad in split_grads if grad is not None]
    if len(split_grads) == 1:
        return split_grads[0]

    # Dealing with IndexedSlices for large-dimensional embedding
    # matrix. The gradient of an embedding matrix is not a tensor,
    # but a tuple-like object named `IndexedSlices`, for this one,
    # we need to take special processings.
    if split_grads[0].__str__().startswith("IndexedSlices"):

        values = tf.divide(tf.concat([grad.values for grad in split_grads], axis=0), len(split_grads))
        indices = tf.concat([grad.indices for grad in split_grads], axis=0)
        dense_shape = split_grads[0].dense_shape

        return tf.IndexedSlices(
            values=values,
            indices=indices,
            dense_shape=dense_shape)

    return tf.divide(tf.add_n(split_grads), len(split_grads))


def update_global_params(variables, global_step, optimizer, grads):
    assert len(grads) == len(variables)
    update_op = optimizer.apply_gradients(
        zip(grads, variables), global_step=global_step)
    return tf.group(update_op)
