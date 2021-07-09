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
''' Wide & Deep structure. '''

import math

from ..tools import tf
from .base import BaseDecoder
from . import util



class WideAndDeepDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 n_wide_features,
                 wide_features,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 scope='cls/seq_relationship',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        hidden_size = input_tensor.shape.as_list()[-1]
        feature_size = wide_features.shape.as_list()[-1]
        with tf.variable_scope('wide'):
            feature_embeddings = tf.get_variable(
                name='feature_embeddings',
                shape=[feature_size + 1, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            wide_output = tf.gather(
                feature_embeddings, wide_features)  # [B, N, H]

        with tf.variable_scope('wide_and_deep'):
            deep_output = tf.expand_dims(input_tensor, -1)  # [B, H, 1]
            attention_scores = tf.matmul(wide_output, deep_output)  # [B, N, 1]
            attention_scores = tf.transpose(attention_scores, [0, 2, 1])  # [B, 1, N]
            attention_scores = tf.multiply(
                attention_scores, 1.0 / math.sqrt(hidden_size))
            feature_mask = tf.cast(
                tf.sequence_mask(n_wide_features, feature_size), tf.float32)    # [B, N]
            feature_mask = tf.expand_dims(feature_mask, 1)  # [B, 1, N]
            attention_scores += (1.0 - feature_mask) * -10000.0
            attention_matrix = tf.nn.softmax(attention_scores, axis=-1)
            attention_output = tf.matmul(attention_matrix, wide_output)  # [B, 1, H]
            attention_output = attention_output[:, 0, :]  # [B, H]
            # attention_output = util.dropout(
            #     attention_output, hidden_dropout_prob)
            input_tensor = util.layer_norm(
                attention_output + input_tensor,
                trainable=trainable)

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

            self._preds['preds'] = tf.argmax(logits, axis=-1)
            self._probs['probs'] = tf.nn.softmax(logits, axis=-1, name='probs')

            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=label_size, dtype=tf.float32)
            per_example_loss = - tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)
            if sample_weight is not None:
                per_example_loss = tf.cast(
                    sample_weight, dtype=tf.float32) * per_example_loss
            thresh = kwargs.get('tsa_thresh')
            if thresh is not None:
                assert isinstance(thresh, float), (
                    '`tsa_thresh` must be a float between 0 and 1.')
                uncertainty = tf.reduce_sum(self._probs['probs'] * tf.log(
                    self._probs['probs']), axis=-1)
                uncertainty /= tf.log(1 / label_size)
                per_example_loss = tf.cast(
                    tf.greater(uncertainty, thresh), dtype=tf.float32) * \
                    per_example_loss

            self._losses['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)
