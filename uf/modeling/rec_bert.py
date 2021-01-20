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
''' Recorrection language modeling. '''

from uf.tools import tf
from .base import BaseDecoder
from .bert import BERTEncoder
from . import util



class RecBERT(BaseDecoder, BERTEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 replace_label_ids,
                 add_label_ids,
                 subtract_label_ids,
                 sample_weight=None,
                 replace_prob=0,
                 add_prob=0,
                 subtract_prob=0,
                 scope='bert',
                 use_tilda_embedding=False,
                 **kwargs):
        super().__init__()

        input_mask = tf.cast(
            tf.not_equal(input_ids, 0), tf.float32)

        shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = shape[0]
        max_seq_length = shape[1]

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):

            # forward once
            hidden = self._bert_forward(
                bert_config,
                input_ids,
                input_mask,
                batch_size,
                max_seq_length,
                tilda_embeddings=tilda_embeddings)

            self.total_loss = 0
            self._lm_forward(
                is_training,
                input_tensor=hidden,
                input_mask=input_mask,
                label_ids=replace_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=replace_prob,
                scope='cls/replace',
                name='replace',
                sample_weight=sample_weight)
            self._lm_forward(
                is_training,
                input_tensor=hidden,
                input_mask=input_mask,
                label_ids=add_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=add_prob,
                scope='cls/add',
                name='add',
                sample_weight=sample_weight)
            self._cls_forward(
                is_training,
                input_tensor=hidden,
                input_mask=input_mask,
                label_ids=subtract_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=subtract_prob,
                scope='cls/subtract',
                name='subtract',
                sample_weight=sample_weight)

    def _bert_forward(self,
                     bert_config,
                     input_ids,
                     input_mask,
                     batch_size,
                     max_seq_length,
                     dtype=tf.float32,
                     trainable=True,
                     tilda_embeddings=None):

        with tf.variable_scope('embeddings'):

            (embedding_output, self.embedding_table) = self.embedding_lookup(
                input_ids=input_ids,
                vocab_size=bert_config.vocab_size,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                embedding_size=bert_config.hidden_size,
                initializer_range=bert_config.initializer_range,
                word_embedding_name='word_embeddings',
                dtype=dtype,
                trainable=trainable,
                tilda_embeddings=tilda_embeddings)

            # Add positional embeddings and token type embeddings
            # layer normalize and perform dropout.
            embedding_output = self.embedding_postprocessor(
                input_tensor=embedding_output,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                hidden_size=bert_config.hidden_size,
                use_token_type=False,
                use_position_embeddings=True,
                position_embedding_name='position_embeddings',
                initializer_range=bert_config.initializer_range,
                max_position_embeddings=\
                    bert_config.max_position_embeddings,
                dropout_prob=bert_config.hidden_dropout_prob,
                dtype=dtype,
                trainable=trainable)

        with tf.variable_scope('encoder'):
            attention_mask = self.create_attention_mask_from_input_mask(
                input_mask, batch_size, max_seq_length, dtype=dtype)

            # stacked transformers
            all_encoder_layers = self.transformer_model(
                input_tensor=embedding_output,
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
                dtype=dtype,
                trainable=trainable)

        return all_encoder_layers[-1]

    def _lm_forward(self,
                    is_training,
                    input_tensor,
                    input_mask,
                    label_ids,
                    bert_config,
                    batch_size,
                    max_seq_length,
                    prob,
                    scope,
                    name,
                    sample_weight=None,
                    hidden_dropout_prob=0.1,
                    initializer_range=0.02):

        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[bert_config.hidden_size, bert_config.hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=True)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[bert_config.hidden_size],
                initializer=tf.zeros_initializer(),
                trainable=True)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            flattened = tf.reshape(
                logits,
                [batch_size * max_seq_length, bert_config.hidden_size])
            logits = tf.matmul(
                flattened, self.embedding_table, transpose_b=True)
            logits = tf.reshape(
                logits, [-1, max_seq_length, bert_config.vocab_size])

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size)
            per_token_loss = -tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)

            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(
                input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_mean(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            if prob != 0:
                self.total_loss += tf.reduce_mean(per_example_loss)
            self.losses[name + '_loss'] = per_example_loss
            self.preds[name + '_preds'] = tf.argmax(logits, axis=-1)

    def _cls_forward(self,
                     is_training,
                     input_tensor,
                     input_mask,
                     label_ids,
                     bert_config,
                     batch_size,
                     max_seq_length,
                     prob,
                     scope,
                     name,
                     sample_weight=None,
                     hidden_dropout_prob=0.1,
                     initializer_range=0.02):

        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[2, bert_config.hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=True)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[2],
                initializer=tf.zeros_initializer(),
                trainable=True)

            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, max_seq_length, 2])

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=2)
            per_token_loss = -tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)

            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(
                input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_mean(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            if prob != 0:
                self.total_loss += tf.reduce_mean(per_example_loss)
            self.losses[name + '_loss'] = per_example_loss
            self.preds[name + '_preds'] = tf.argmax(logits, axis=-1)
