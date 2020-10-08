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
''' Dilated language modeling. '''

from uf.tools import tf
from .base import BaseDecoder
from .bert import BERTEncoder
from . import util



class DLM(BaseDecoder, BERTEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 label_ids,
                 max_seq_length,
                 spad_id=1,
                 loop=3,
                 sample_weight=None,
                 scope='dilated',
                 use_tilda_embedding=False,
                 **kwargs):
        super().__init__()

        shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = shape[0]
        dilated_seq_length = shape[1]

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):

            # forward once
            if is_training:
                logits = self._bert_forward(
                    bert_config,
                    input_ids,
                    input_mask,
                    batch_size,
                    dilated_seq_length,
                    tilda_embeddings=tilda_embeddings)

                self.preds['LM'] = tf.argmax(logits, axis=-1)

                # LM loss
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                one_hot_labels = tf.one_hot(
                    label_ids, depth=bert_config.vocab_size)
                per_token_loss = -tf.reduce_sum(
                    one_hot_labels * log_probs, axis=-1)
                label_mask = tf.cast(input_mask, tf.float32)
                per_example_loss = \
                    tf.reduce_sum(per_token_loss * label_mask, axis=-1) / \
                    tf.reduce_sum(label_mask, axis=-1)
                if sample_weight is not None:
                    per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

                self.total_loss = tf.reduce_mean(per_example_loss)
                self.losses['LM'] = per_example_loss

            # forward loop
            else:
                def _forward(dilated_ids, dilated_mask):

                    logits = self._bert_forward(
                        bert_config,
                        dilated_ids,
                        dilated_mask,
                        batch_size,
                        dilated_seq_length,
                        tilda_embeddings=tilda_embeddings)
                    output_ids = tf.argmax(logits, axis=-1)
                    output_ids = tf.cast(output_ids, dtype=tf.int32)

                    equal_zero = tf.cast(tf.equal(output_ids, 0), tf.int32)
                    equal_zero = tf.reduce_sum(equal_zero, axis=-1)
                    right_pad = spad_id * tf.sequence_mask(
                        equal_zero, dilated_seq_length, dtype=tf.int32)

                    paded = tf.concat([output_ids, right_pad], axis=-1)
                    flattened_padded = tf.reshape(paded, [-1])
                    is_valid = tf.cast(
                        tf.greater(flattened_padded, 0), dtype=tf.int32)
                    flattened_valid = tf.boolean_mask(
                        flattened_padded, is_valid)
                    valid = tf.reshape(
                        flattened_valid, [batch_size, dilated_seq_length])
                    cutted_valid = valid[:, :max_seq_length]

                    nonpad_mask = tf.cast(
                        tf.not_equal(cutted_valid, spad_id), dtype=tf.int32)
                    output_ids = cutted_valid * nonpad_mask

                    reshaped = tf.reshape(
                        output_ids, [batch_size, max_seq_length, 1])
                    concatenated = tf.concat(
                        [reshaped, tf.zeros_like(reshaped)], axis=-1)
                    dilated_ids = tf.reshape(
                        concatenated, [batch_size, max_seq_length * 2])

                    input_mask = tf.reduce_sum(nonpad_mask, axis=-1)
                    dilated_mask = tf.sequence_mask(
                        input_mask, dilated_seq_length, dtype=tf.int32)

                    return dilated_ids, dilated_mask

                dilated_ids = input_ids
                dilated_mask = input_mask
                for _ in range(loop):
                    dilated_ids, dilated_mask = _forward(
                        dilated_ids, dilated_mask)

                self.preds['LM'] = dilated_ids

    def _bert_forward(self,
                     bert_config,
                     input_ids,
                     input_mask,
                     batch_size,
                     dilated_seq_length,
                     dtype=tf.float32,
                     trainable=True,
                     tilda_embeddings=None):

        with tf.variable_scope('embeddings'):

            (embedding_output, embedding_table) = self.embedding_lookup(
                input_ids=input_ids,
                vocab_size=bert_config.vocab_size,
                batch_size=batch_size,
                max_seq_length=dilated_seq_length,
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
                max_seq_length=dilated_seq_length,
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
                input_mask, batch_size, dilated_seq_length, dtype=dtype)

            # stacked transformers
            all_encoder_layers = self.transformer_model(
                input_tensor=embedding_output,
                batch_size=batch_size,
                max_seq_length=dilated_seq_length,
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

        flattened = tf.reshape(
            all_encoder_layers[-1],
            [batch_size * dilated_seq_length, bert_config.hidden_size])
        logits = tf.matmul(flattened, embedding_table, transpose_b=True)
        logits = tf.reshape(
            logits, [-1, dilated_seq_length, bert_config.vocab_size])

        return logits