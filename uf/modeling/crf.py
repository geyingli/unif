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
''' Conditional Random Field (CRF). '''


from uf.tools import tf, contrib
from .base import BaseDecoder
from . import util


class CRFDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 input_mask,
                 label_ids,
                 label_size=5,
                 sample_weight=None,
                 scope='cls/sequence',
                 name='',
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

            with tf.variable_scope('crf'):
                input_length = tf.reduce_sum(input_mask, axis=-1)
                per_example_loss, transition_matrix = \
                    contrib.crf.crf_log_likelihood(
                        inputs=logits,
                        tag_indices=label_ids,
                        sequence_lengths=input_length)
                per_example_loss = - per_example_loss
                if sample_weight is not None:
                    per_example_loss *= tf.cast(
                        sample_weight, dtype=tf.float32)
                self.total_loss = tf.reduce_mean(per_example_loss)
                self.losses[name] = per_example_loss
                self.preds[name] = tf.argmax(logits, axis=-1)
                self.probs['logits'] = logits
                self.probs['transition_matrix'] = transition_matrix
