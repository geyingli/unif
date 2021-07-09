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
''' SemBERT decoder. '''

import copy

from ..tools import tf
from .base import BaseEncoder
from .bert import BERTEncoder
from . import util



class StockBERTEncoder(BERTEncoder, BaseEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_values,
                 input_mask,
                 scope='stock_bert',
                 drop_pooler=False,
                 trainable=True,
                 **kwargs):

        bert_config = copy.deepcopy(bert_config)
        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(input_values, expected_rank=3)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1] + 1

        with tf.variable_scope(scope):
            with tf.variable_scope('embeddings'):

                self.embedding_output = self.embedding_preprocessor(
                    input_values=input_values,
                    batch_size=batch_size,
                    embedding_size=bert_config.hidden_size,
                    initializer_range=bert_config.initializer_range,
                    name='cls_embedding',
                    trainable=trainable)

                # Add positional embeddings and token type embeddings
                # layer normalize and perform dropout.
                self.embedding_output = self.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    hidden_size=bert_config.hidden_size,
                    use_token_type=False,
                    segment_ids=None,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=bert_config.initializer_range,
                    max_position_embeddings=\
                        bert_config.max_position_embeddings,
                    dropout_prob=bert_config.hidden_dropout_prob,
                    trainable=trainable)

            with tf.variable_scope('encoder'):
                attention_mask = self.create_attention_mask_from_input_mask(
                    input_mask, batch_size, max_seq_length)

                # stacked transformers
                self.all_encoder_layers = self.transformer_model(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    attention_mask=attention_mask,
                    hidden_size=bert_config.hidden_size,
                    num_hidden_layers=bert_config.num_hidden_layers,
                    num_attention_heads=bert_config.num_attention_heads,
                    intermediate_size=bert_config.intermediate_size,
                    intermediate_act_fn=util.get_activation(
                        bert_config.hidden_act),
                    hidden_dropout_prob=bert_config.hidden_dropout_prob,
                    attention_probs_dropout_prob=\
                    bert_config.attention_probs_dropout_prob,
                    initializer_range=bert_config.initializer_range,
                    trainable=trainable)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope('pooler'):
                first_token_tensor = self.sequence_output[:, 0, :]

                # trick: ignore the fully connected layer
                if drop_pooler:
                    self.pooled_output = first_token_tensor
                else:
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        bert_config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range),
                        trainable=trainable)

    def embedding_preprocessor(self,
                               input_values,
                               batch_size=None,
                               embedding_size=128,
                               initializer_range=0.02,
                               name='cls_embedding',
                               dtype=tf.float32,
                               trainable=True):

        with tf.variable_scope(name):
            input_values = util.layer_norm(
                input_values,
                trainable=trainable)
            linear_output = tf.layers.dense(
                input_values,
                embedding_size,
                activation=None,
                name='dense',
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable)

            cls_embedding = tf.get_variable(
                name='cls',
                shape=[1, 1, embedding_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)
            cls_output = tf.tile(cls_embedding, [batch_size, 1, 1])

        output = tf.concat([cls_output, linear_output], axis=1)
        return output
