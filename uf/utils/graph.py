# coding:=utf-8
# Copyright 2021 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    tf.logging.info('Build graph with %s parameters '
                    '(among which %s are trainable)'
                    % (format(int(n_global), ','),
                       format(int(n_trainable), ',')))


def average_n_grads(split_grads):
    split_grads = [grad for grad in split_grads if grad is not None]
    if len(split_grads) == 1:
        return split_grads[0]

    # Dealing with IndexedSlices for large-dimensional embedding
    # matrix. The gradient of an embedding matrix is not a tensor,
    # but a tuple-like object named `IndexedSlices`, for this one,
    # we need to take special processings.
    if split_grads[0].__str__().startswith('IndexedSlices'):

        values = tf.concat([grad.values for grad in split_grads], axis=0)
        indices = tf.concat([grad.indices for grad in split_grads], axis=0)
        dense_shape = split_grads[0].dense_shape

        return tf.IndexedSlices(
            values=values,
            indices=indices,
            dense_shape=dense_shape)

    return tf.divide(tf.add_n(split_grads), len(split_grads))


def update_global_params(variables, global_step, optimizer, grads):
    update_op = optimizer.apply_gradients(
        zip(grads, variables), global_step=global_step)
    return tf.group(update_op)
