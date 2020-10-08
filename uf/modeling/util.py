# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
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
''' Commonly-used modeling methods. '''

import numpy as np

from uf.tools import tf
from uf.tools import contrib


def gelu(num):
    ''' Gaussian Error Linear Unit, a smoother version of the RELU.
    paper: https://arxiv.org/abs/1606.08415 '''
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
                                (num + 0.044715 * tf.pow(num, 3)))))
    return num * cdf


def get_activation(activation_string):
    ''' Returns activation function given string. '''
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == 'linear':
        return None
    if act == 'relu':
        return tf.nn.relu
    if act == 'gelu':
        return gelu
    if act == 'tanh':
        return tf.tanh
    raise ValueError('Unsupported activation: %s' % act)


def dropout(input_tensor, dropout_prob):
    ''' A more intuitive dropout function. '''
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    try:
        output = tf.nn.dropout(input_tensor, keep_prob=1.0 - dropout_prob)
    except:
        output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def layer_norm(input_tensor, name=None, trainable=True):
    ''' Runs layer normalization on the last dimension of the tensor. '''
    return contrib.layers.layer_norm(
        inputs=input_tensor,
        begin_norm_axis=-1,
        begin_params_axis=-1,
        scope=name,
        trainable=trainable)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None,
                           trainable=True):
    ''' Runs layer normalization followed by dropout. '''
    output_tensor = layer_norm(input_tensor, name=name, trainable=trainable)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def create_initializer(initializer_range=0.02):
    '''Creates a truncated_normal_initializer with the given range.'''
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor, expected_rank=None, name=None):
    '''Returns a list of the shape of tensor, preferring static dimensions.'''
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    '''Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).'''
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError(
            'Input tensor must have at least rank 2. Shape = %s'
            % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list,
                        original_shape=None):
    '''Reshapes a rank 2 tensor back to its original rank >= 2 tensor.'''
    if len(orig_shape_list) == 2:
        return output_tensor

    if not original_shape:
        original_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = original_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    '''Raises an exception if the tensor rank is not of the expected rank.'''
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for rank in expected_rank:
            expected_rank_dict[rank] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            'For the tensor %s in scope %s, the actual rank '
            '%d (shape = %s) is not equal to the expected rank %s'
            % (name,
               scope_name,
               actual_rank,
               str(tensor.shape),
               str(expected_rank)))
