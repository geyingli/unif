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
''' Base class for encoder and decoder. '''

import collections

from ..tools import tf
from . import util


class BaseEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def get_pooled_output(self):
        raise NotImplementedError()

    def get_sequence_output(self):
        raise NotImplementedError()


class BaseDecoder:
    def __init__(self, *args, **kwargs):

        # scalar of total loss
        self.total_loss = None

        # supervised tensors of each example
        self._tensors = collections.OrderedDict()

    def get_forward_outputs(self):
        return (self.total_loss, self._tensors)


class CLSDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 scope='cls',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            self._tensors['preds'] = tf.argmax(logits, axis=-1, name='preds')
            self._tensors['probs'] = tf.nn.softmax(
                logits, axis=-1, name='probs')

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=label_size, dtype=tf.float32)
            per_example_loss = - tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)
            if sample_weight is not None:
                per_example_loss = tf.cast(
                    sample_weight, dtype=tf.float32) * per_example_loss
            thresh = kwargs.get('conf_thresh')
            if thresh is not None:
                assert isinstance(thresh, float), (
                    '`conf_thresh` must be a float between 0 and 1.')
                largest_prob = tf.reduce_max(tf.exp(log_probs), axis=-1)
                per_example_loss = tf.cast(
                    tf.less(largest_prob, thresh), dtype=tf.float32) * \
                    per_example_loss

            self._tensors['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)


class RegDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 label_floats,
                 label_size=2,
                 sample_weight=None,
                 scope='reg',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            intermediate_output = tf.layers.dense(
                input_tensor,
                label_size * 4,
                use_bias=False,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable,
            )
            preds = tf.layers.dense(
                intermediate_output,
                label_size,
                use_bias=False,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable,
                name='preds',
            )

            self._tensors['preds'] = preds

            per_example_loss = tf.reduce_sum(
                tf.square(label_floats - preds), axis=-1)
            if sample_weight is not None:
                per_example_loss = tf.cast(
                    sample_weight, dtype=tf.float32) * per_example_loss

            self._tensors['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)


class BinaryCLSDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 label_weight=None,
                 scope='cls/seq_relationship',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probs = tf.nn.sigmoid(logits, name='probs')

            self._tensors['probs'] = probs
            self._tensors['preds'] = tf.greater(probs, 0.5, name='preds')

            per_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=tf.cast(label_ids, dtype=tf.float32))
            if label_weight is not None:
                label_weight = tf.constant(label_weight, dtype=tf.float32)
                label_weight = tf.reshape(label_weight, [1, label_size])
                per_label_loss *= label_weight
            per_example_loss = tf.reduce_sum(per_label_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self._tensors['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)


class SeqCLSDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 input_mask,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 scope='cls/sequence',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        shape = input_tensor.shape.as_list()
        seq_length = shape[-2]
        hidden_size = shape[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)

            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, seq_length, label_size])

            self._tensors['preds'] = tf.argmax(logits, axis=-1, name='preds')
            self._tensors['probs'] = tf.nn.softmax(
                logits, axis=-1, name='probs')

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=label_size, dtype=tf.float32)
            per_token_loss = - tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)

            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(
                input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.cast(
                    sample_weight, dtype=tf.float32)

            self._tensors['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)


class MRCDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 label_ids,
                 sample_weight=None,
                 scope='mrc',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        seq_length = input_tensor.shape.as_list()[-2]
        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[2, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[2],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)

            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, seq_length, 2])
            logits = tf.transpose(logits, [0, 2, 1])
            probs = tf.nn.softmax(logits, axis=-1, name='probs')
            self._tensors['probs'] = probs
            self._tensors['preds'] = tf.argmax(logits, axis=-1, name='preds')

            start_one_hot_labels = tf.one_hot(
                label_ids[:, 0], depth=seq_length, dtype=tf.float32)
            end_one_hot_labels = tf.one_hot(
                label_ids[:, 1], depth=seq_length, dtype=tf.float32)
            start_log_probs = tf.nn.log_softmax(logits[:, 0, :], axis=-1)
            end_log_probs = tf.nn.log_softmax(logits[:, 1, :], axis=-1)
            per_example_loss = (
                - 0.5 * tf.reduce_sum(
                    start_one_hot_labels * start_log_probs, axis=-1)
                - 0.5 * tf.reduce_sum(
                    end_one_hot_labels * end_log_probs, axis=-1))
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.total_loss = tf.reduce_mean(per_example_loss)
            self._tensors['losses'] = per_example_loss
