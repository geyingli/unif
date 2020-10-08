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
''' Bidirectional Encoder Representations from Transformers (BERT).
  Code revised from Google's implementation.
  See `https://github.com/google-research/bert`.
'''

import math
import copy
import json

from uf.tools import tf
from .base import BaseEncoder, BaseDecoder
from . import util


class BERTEncoder(BaseEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 scope='bert',
                 dtype=tf.float32,
                 use_tilda_embedding=False,
                 drop_pooler=False,
                 trainable=True,
                 **kwargs):

        bert_config = copy.deepcopy(bert_config)
        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):
            with tf.variable_scope('embeddings'):

                (self.embedding_output, self.embedding_table) = \
                    self.embedding_lookup(
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
                self.embedding_output = self.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    hidden_size=bert_config.hidden_size,
                    use_token_type=True,
                    segment_ids=segment_ids,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
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
                    dtype=dtype,
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

    def get_pooled_output(self):
        ''' Returns a tensor with shape [batch_size, hidden_size]. '''
        return self.pooled_output

    def get_sequence_output(self):
        ''' Returns a tensor with shape
        [batch_size, max_seq_length, hidden_size]. '''
        return self.sequence_output

    def get_embedding_table(self):
        return self.embedding_table

    def get_embedding_output(self):
        return self.embedding_output

    def get_attention_scores(self):
        return self.attention_scores

    def embedding_lookup(self,
                         input_ids,
                         vocab_size,
                         batch_size,
                         max_seq_length,
                         embedding_size=128,
                         initializer_range=0.02,
                         word_embedding_name='word_embeddings',
                         dtype=tf.float32,
                         trainable=True,
                         tilda_embeddings=None):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        if tilda_embeddings is not None:
            embedding_table = tilda_embeddings
        else:
            embedding_table = tf.get_variable(
                name=word_embedding_name,
                shape=[vocab_size, embedding_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)

        flat_input_ids = tf.reshape(input_ids, [-1])
        output = tf.gather(
            embedding_table, flat_input_ids, name='embedding_look_up')
        output = tf.reshape(
            output, [batch_size, max_seq_length, embedding_size])

        return (output, embedding_table)

    def embedding_postprocessor(self,
                                input_tensor,
                                batch_size,
                                max_seq_length,
                                hidden_size,
                                use_token_type=False,
                                segment_ids=None,
                                token_type_vocab_size=16,
                                token_type_embedding_name=\
                                    'token_type_embeddings',
                                use_position_embeddings=True,
                                position_embedding_name='position_embeddings',
                                initializer_range=0.02,
                                max_position_embeddings=512,
                                dropout_prob=0.1,
                                dtype=tf.float32,
                                trainable=True):
        output = input_tensor

        if use_token_type:
            if segment_ids is None:
                raise ValueError(
                    'segment_ids must be specified if use_token_type is True.')
            token_type_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)

            # This vocab will be small so we always do one-hot here,
            # since it is always faster for a small vocabulary.
            flat_segment_ids = tf.reshape(segment_ids, [-1])
            one_hot_ids = tf.one_hot(
                flat_segment_ids, depth=token_type_vocab_size, dtype=dtype)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(
                token_type_embeddings,
                [batch_size, max_seq_length, hidden_size])
            output += token_type_embeddings

        if use_position_embeddings:
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, hidden_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)
            position_embeddings = tf.slice(
                full_position_embeddings, [0, 0], [max_seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant
            # (max_seq_length and hidden_size), so we broadcast
            # among the first dimensions, which is typically
            # just the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([max_seq_length, hidden_size])
            position_embeddings = tf.reshape(
                position_embeddings, position_broadcast_shape)
            output += position_embeddings

        output = util.layer_norm_and_dropout(
            output, dropout_prob, trainable=trainable)
        return output

    def create_attention_mask_from_input_mask(self,
                                              to_mask,
                                              batch_size,
                                              max_seq_length,
                                              dtype=tf.float32):
        to_mask = tf.cast(tf.reshape(
            to_mask, [batch_size, 1, max_seq_length]), dtype=dtype)
        broadcast_ones = tf.ones(
            shape=[batch_size, max_seq_length, 1], dtype=dtype)
        mask = broadcast_ones * to_mask
        return mask

    def attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_max_seq_length=None,
                        to_max_seq_length=None,
                        dtype=tf.float32,
                        trainable=True):

        def transpose_for_scores(input_tensor, batch_size,
                                 num_attention_heads, max_seq_length, width):
            output_tensor = tf.reshape(
                input_tensor,
                [batch_size, max_seq_length, num_attention_heads, width])
            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = from_tensor sequence length
        #   T = to_tensor sequence length
        #   N = num_attention_heads
        #   H = size_per_head

        from_tensor_2d = util.reshape_to_matrix(from_tensor)
        to_tensor_2d = util.reshape_to_matrix(to_tensor)

        # query_layer = [B*F, N*H]
        query_layer = tf.layers.dense(
            from_tensor_2d,
            num_attention_heads * size_per_head,
            activation=query_act,
            name='query',
            kernel_initializer=util.create_initializer(initializer_range),
            trainable=trainable)

        # key_layer = [B*T, N*H]
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=key_act,
            name='key',
            kernel_initializer=util.create_initializer(initializer_range),
            trainable=trainable)

        # value_layer = [B*T, N*H]
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=value_act,
            name='value',
            kernel_initializer=util.create_initializer(initializer_range),
            trainable=trainable)

        # query_layer = [B, N, F, H]
        query_layer = transpose_for_scores(
            query_layer, batch_size, num_attention_heads,
            from_max_seq_length, size_per_head)

        # key_layer = [B, N, T, H]
        key_layer = transpose_for_scores(
            key_layer, batch_size, num_attention_heads,
            to_max_seq_length, size_per_head)

        # Take the dot product between 'query' and 'key' to get the raw
        # attention scores.
        # attention_scores = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(
            attention_scores, 1.0 / math.sqrt(float(size_per_head)))

        if attention_mask is not None:

            # attention_mask = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(attention_mask, dtype)) * -10000.0
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # attention_probs = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to,
        # which might seem a bit unusual, but is taken from the original
        # Transformer paper.
        attention_probs = util.dropout(
            attention_probs, attention_probs_dropout_prob)

        # value_layer = [B, T, N, H]
        value_layer = tf.reshape(
            value_layer, [batch_size, to_max_seq_length,
                          num_attention_heads, size_per_head])

        # value_layer = [B, N, T, H]
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

        # context_layer = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value_layer)

        # context_layer = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # context_layer = [B*F, N*H]
            context_layer = tf.reshape(
                context_layer, [batch_size * from_max_seq_length,
                                num_attention_heads * size_per_head])
        else:
            # context_layer = [B, F, N*H]
            context_layer = tf.reshape(
                context_layer, [batch_size, from_max_seq_length,
                                num_attention_heads * size_per_head])

        return (context_layer, attention_scores)

    def transformer_model(self,
                          input_tensor,
                          batch_size,
                          max_seq_length,
                          attention_mask=None,
                          hidden_size=768,
                          num_hidden_layers=12,
                          num_attention_heads=12,
                          intermediate_size=3072,
                          intermediate_act_fn=util.gelu,
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          initializer_range=0.02,
                          dtype=tf.float32,
                          trainable=True):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number '
                'of attention heads (%d)'
                % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        prev_output = util.reshape_to_matrix(input_tensor)

        self.attention_scores = []
        all_layer_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('layer_%d' % layer_idx):
                layer_input = prev_output

                def _build_forward(layer_input):
                    with tf.variable_scope('attention'):
                        attention_heads = []
                        with tf.variable_scope('self'):
                            (attention_head, attention_scores) = \
                                self.attention_layer(
                                    from_tensor=layer_input,
                                    to_tensor=layer_input,
                                    attention_mask=attention_mask,
                                    num_attention_heads=num_attention_heads,
                                    size_per_head=attention_head_size,
                                    attention_probs_dropout_prob=\
                                        attention_probs_dropout_prob,
                                    initializer_range=initializer_range,
                                    do_return_2d_tensor=True,
                                    batch_size=batch_size,
                                    from_max_seq_length=max_seq_length,
                                    to_max_seq_length=max_seq_length,
                                    dtype=dtype,
                                    trainable=trainable)
                            attention_heads.append(attention_head)
                            self.attention_scores.append(attention_scores)

                        attention_output = None
                        if len(attention_heads) == 1:
                            attention_output = attention_heads[0]
                        else:
                            attention_output = tf.concat(
                                attention_heads, axis=-1)

                        with tf.variable_scope('output'):
                            attention_output = tf.layers.dense(
                                attention_output,
                                hidden_size,
                                kernel_initializer=util.create_initializer(
                                    initializer_range),
                                trainable=trainable)
                            attention_output = util.dropout(
                                attention_output, hidden_dropout_prob)
                            attention_output = util.layer_norm(
                                attention_output + layer_input,
                                trainable=trainable)

                    # The activation is only applied to the `intermediate`
                    # hidden layer.
                    with tf.variable_scope('intermediate'):
                        intermediate_output = tf.layers.dense(
                            attention_output,
                            intermediate_size,
                            activation=intermediate_act_fn,
                            kernel_initializer=util.create_initializer(
                                initializer_range),
                            trainable=trainable)

                    # Down-project back to hidden_size then add the residual.
                    with tf.variable_scope('output'):
                        layer_output = tf.layers.dense(
                            intermediate_output,
                            hidden_size,
                            kernel_initializer=util.create_initializer(
                                initializer_range),
                            trainable=trainable)
                        layer_output = util.dropout(
                            layer_output, hidden_dropout_prob)
                        layer_output = util.layer_norm(
                            layer_output + attention_output,
                            trainable=trainable)

                    return layer_output

                layer_output = _build_forward(layer_input)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

        original_shape = [batch_size * max_seq_length, hidden_size]
        input_shape = [batch_size, max_seq_length, hidden_size]

        final_all_layer_outputs = []
        for layer_output in all_layer_outputs:
            final_output = util.reshape_from_matrix(
                layer_output, input_shape, original_shape=original_shape)
            final_all_layer_outputs.append(final_output)
        return final_all_layer_outputs



class BERTDecoder(BaseDecoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 encoder,
                 masked_lm_positions,
                 masked_lm_ids,
                 masked_lm_weights,
                 next_sentence_labels=None,
                 sample_weight=None,
                 scope_lm='cls/predictions',
                 scope_cls='cls/seq_relationship',
                 name='',
                 trainable=True,
                 **kwargs):
        super(BERTDecoder, self).__init__(**kwargs)

        def gather_indexes(sequence_tensor, positions):
            sequence_shape = util.get_shape_list(sequence_tensor, 3)
            batch_size = sequence_shape[0]
            seq_length = sequence_shape[1]
            width = sequence_shape[2]

            flat_offsets = tf.reshape(
                tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
            flat_positions = tf.reshape(positions + flat_offsets, [-1])
            flat_sequence_tensor = tf.reshape(
                sequence_tensor, [batch_size * seq_length, width])
            output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
            return output_tensor

        scalar_losses = []

        # masked language modeling
        input_tensor = gather_indexes(
            encoder.get_sequence_output(), masked_lm_positions)
        with tf.variable_scope(scope_lm):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=bert_config.hidden_size,
                    activation=util.get_activation(bert_config.hidden_act),
                    kernel_initializer=util.create_initializer(
                        bert_config.initializer_range))
                input_tensor = util.layer_norm(input_tensor)
            output_bias = tf.get_variable(
                'output_bias', shape=[bert_config.vocab_size],
                initializer=tf.zeros_initializer())

            logits = tf.matmul(
                input_tensor, encoder.get_embedding_table(), transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probs = tf.nn.softmax(logits, axis=-1, name='MLM_probs')
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(masked_lm_ids, [-1])
            if sample_weight is not None:
                sample_weight = tf.expand_dims(
                    tf.cast(sample_weight, dtype=tf.float32), axis=-1)
                masked_lm_weights *= sample_weight
            label_weights = tf.reshape(masked_lm_weights, [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
            per_example_loss = - tf.reduce_sum(
                log_probs * one_hot_labels, axis=[-1])
            per_example_loss = label_weights * per_example_loss

            numerator = tf.reduce_sum(per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            loss = numerator / denominator

            scalar_losses.append(loss)
            self.losses['MLM'] = per_example_loss
            self.preds['MLM'] = tf.argmax(probs, axis=-1)

        # next sentence prediction
        if next_sentence_labels is not None:
            with tf.variable_scope(scope_cls):
                output_weights = tf.get_variable(
                    'output_weights',
                    shape=[2, bert_config.hidden_size],
                    initializer=util.create_initializer(
                        bert_config.initializer_range))
                output_bias = tf.get_variable(
                    'output_bias', shape=[2],
                    initializer=tf.zeros_initializer())

                logits = tf.matmul(encoder.get_pooled_output(),
                                   output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                probs = tf.nn.softmax(logits, axis=-1, name='probs')
                log_probs = tf.nn.log_softmax(logits, axis=-1)

                labels = tf.reshape(next_sentence_labels, [-1])
                one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
                per_example_loss = -tf.reduce_sum(
                    one_hot_labels * log_probs, axis=-1)
                if sample_weight is not None:
                    per_example_loss = (
                        tf.cast(sample_weight, dtype=tf.float32) *
                        per_example_loss)
                loss = tf.reduce_mean(per_example_loss)

                scalar_losses.append(loss)
                self.losses[name] = per_example_loss
                self.probs[name] = probs
                self.preds[name] = tf.argmax(probs, axis=-1)

        self.total_loss = tf.add_n(scalar_losses)


class BERTConfig:
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_obj):
        config = BERTConfig(vocab_size=None)
        for key, value in json_obj.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        ''' Load from json file. '''
        with open(json_file) as json_fp:
            return cls.from_dict(json.load(json_fp))

    def to_json_file(self, json_file):
        ''' Write into json file. '''
        with open(json_file, 'w') as json_fp:
            return json.dump(self.__dict__, json_fp, indent=2)