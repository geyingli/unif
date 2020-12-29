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
''' SemBERT decoder. '''

from uf.tools import tf
from .base import BaseDecoder
from .bert import BERTEncoder
from . import util



class SemBERTDecoder(BaseDecoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_tensor,
                 input_mask,
                 sem_features,
                 label_ids,
                 max_seq_length,
                 feature_size,
                 label_size=2,
                 sample_weight=None,
                 scope='cls/seq_relationship',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        input_shape = util.get_shape_list(input_tensor)
        batch_size = input_shape[0]
        hidden_size = input_shape[-1]
        with tf.variable_scope('sem'):
            feature_embeddings = tf.get_variable(
                name='feature_embeddings',
                shape=[feature_size + 3, hidden_size],  # for [PAD], [CLS], [SEP]
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            sem_output = tf.gather(
                feature_embeddings, sem_features)  # [B, N, H]

            attention_heads = []
            with tf.variable_scope('self'):
                attention_mask = BERTEncoder.create_attention_mask_from_input_mask(
                    input_mask, batch_size, max_seq_length)
                (attention_head, _) = BERTEncoder.attention_layer(
                    from_tensor=sem_output,
                    to_tensor=sem_output,
                    attention_mask=attention_mask,
                    num_attention_heads=bert_config.num_attention_heads,
                    size_per_head=(hidden_size // bert_config.num_attention_heads),
                    attention_probs_dropout_prob=hidden_dropout_prob if is_training else 0.0,
                    initializer_range=initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_max_seq_length=max_seq_length,
                    to_max_seq_length=max_seq_length,
                    trainable=trainable)
                attention_heads.append(attention_head)

            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)

            attention_output = attention_output[:, 0, :]  # [B, H]
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

            self.preds['preds'] = tf.argmax(logits, axis=-1)
            self.probs['probs'] = tf.nn.softmax(logits, axis=-1, name='probs')

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
                uncertainty = tf.reduce_sum(self.probs['probs'] * tf.log(
                    self.probs['probs']), axis=-1)
                uncertainty /= tf.log(1 / label_size)
                per_example_loss = tf.cast(
                    tf.greater(uncertainty, thresh), dtype=tf.float32) * \
                    per_example_loss

            self.losses['losses'] = per_example_loss
            self.total_loss = tf.reduce_mean(per_example_loss)
