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
                 rep_label_ids,
                 add_label_ids,
                 del_label_ids,
                 sample_weight=None,
                 rep_prob=0,
                 add_prob=0,
                 del_prob=0,
                 scope='bert',
                 use_tilda_embedding=False,
                 **kwargs):
        super().__init__()

        input_mask = tf.cast(
            tf.not_equal(input_ids, 0), tf.float32)

        shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = shape[0]
        max_seq_length = shape[1]

        if is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

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
                label_ids=rep_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=rep_prob,
                scope='cls/rep',
                name='rep',
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
                label_ids=del_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=del_prob,
                scope='cls/del',
                name='del',
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

            with tf.variable_scope('verifier'):
                logits = tf.layers.dense(
                    input_tensor, 2,
                    kernel_initializer=util.create_initializer(
                        bert_config.initializer_range),
                    trainable=True)
                verifier_label_ids = tf.cast(
                    tf.greater(label_ids, 0), tf.int32)

                # loss
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                one_hot_labels = tf.one_hot(
                    verifier_label_ids, depth=2)
                per_token_loss = -tf.reduce_sum(
                    one_hot_labels * log_probs, axis=-1)

                input_mask = tf.cast(input_mask, tf.float32)
                per_token_loss *= input_mask / tf.reduce_sum(
                    input_mask, keepdims=True, axis=-1)
                per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
                if sample_weight is not None:
                    per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

                if prob != 0:
                    self.total_loss += tf.reduce_mean(per_example_loss)
                verifier_loss = per_example_loss
                verifier_preds = tf.argmax(logits, axis=-1)

            with tf.variable_scope('prediction'):

                with tf.variable_scope('intermediate'):
                    logits = tf.layers.dense(
                        input_tensor, bert_config.hidden_size * 4,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range),
                        activation=util.gelu,
                        trainable=True)
                with tf.variable_scope('output'):
                    logits = tf.layers.dense(
                        logits, bert_config.hidden_size,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range),
                        trainable=True)

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

                input_mask *= tf.cast(verifier_preds, tf.float32)
                per_token_loss *= input_mask / (tf.reduce_sum(
                    input_mask, keepdims=True, axis=-1) + 1e-6)
                per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
                if sample_weight is not None:
                    per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

                if prob != 0:
                    self.total_loss += tf.reduce_mean(per_example_loss)
                self.losses[name + '_loss'] = verifier_loss
                self.preds[name + '_preds'] = \
                    tf.argmax(logits, axis=-1) * verifier_preds

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
            logits = tf.layers.dense(
                input_tensor, 2,
                kernel_initializer=util.create_initializer(
                    bert_config.initializer_range),
                trainable=True)

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(
                label_ids, depth=2)
            per_token_loss = -tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)

            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(
                input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            if prob != 0:
                self.total_loss += tf.reduce_mean(per_example_loss)
            self.losses[name + '_loss'] = per_example_loss
            self.preds[name + '_preds'] = tf.argmax(logits, axis=-1)
