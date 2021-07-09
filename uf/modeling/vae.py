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
''' The text generation model VAE we utilize in clustering, feature extraction
and negative sample generation. '''

from uf.tools import tf
from .base import BaseDecoder
from .bert import BERTEncoder
from . import util


class VAE(BaseDecoder, BERTEncoder):
    def __init__(self,
                 vocab_size,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 sample_weight=None,
                 reduced_size=64,
                 topic_size=1024,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 bias=0,
                 scope='vae',
                 trainable=True,
                 **kwargs):
        super().__init__()

        # freeze parameters
        config = Config(
            vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        use_tilda_embedding=kwargs.get('use_tilda_embedding')
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):
            with tf.variable_scope('embeddings'):

                (self.embedding_output, self.embedding_table) = \
                    self.embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=config.vocab_size,
                        batch_size=batch_size,
                        max_seq_length=seq_length,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name='word_embeddings',
                        tilda_embeddings=tilda_embeddings,
                        trainable=trainable)
                self.embedding_output = self.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=seq_length,
                    hidden_size=config.hidden_size,
                    use_token_type=True,
                    segment_ids=segment_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    trainable=trainable)

            with tf.variable_scope('encoder'):

                # stacked transformer
                attention_mask = self.create_attention_mask_from_input_mask(
                    input_mask, batch_size, seq_length)
                self.all_encoder_layers = self.transformer_model(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=seq_length,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=util.get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=\
                        config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    trainable=trainable)

                # projection
                with tf.variable_scope('projection'):
                    transformer_output = tf.layers.dense(
                        self.all_encoder_layers[-1],
                        reduced_size,
                        activation=util.gelu,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=config.initializer_range),
                        trainable=trainable)
                    transformer_output = tf.reshape(
                        transformer_output, [batch_size, -1])
                    input_length = tf.reduce_sum(input_mask, axis=-1)
                    input_length = tf.cast(input_length, tf.float32)
                    input_length_1d = tf.reshape(input_length, [batch_size])
                    input_length_2d = tf.reshape(input_length, [batch_size, 1])

                    broadcast_mask = tf.sequence_mask(
                        tf.multiply(input_length_1d, reduced_size),
                        seq_length * reduced_size,
                        dtype=tf.float32)
                    broadcast_mask = tf.multiply(
                        broadcast_mask, seq_length / input_length_2d)
                    transformer_output *= broadcast_mask

                    # latent space
                    miu = tf.layers.dense(
                        transformer_output,
                        topic_size,
                        activation='tanh',
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=config.initializer_range),
                        name='miu',
                        trainable=trainable)
                    sigma = tf.layers.dense(
                        transformer_output,
                        topic_size,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=config.initializer_range),
                        name='sigma',
                        trainable=trainable)
                    self._probs['miu'] = miu
                    self._probs['sigma'] = sigma

            with tf.variable_scope('decoder'):
                with tf.variable_scope('projection'):

                    # reparametarization
                    if is_training:
                        noise = tf.random_normal([batch_size, topic_size])
                    else:
                        noise = tf.random_uniform(
                            [batch_size, topic_size],
                            minval=-bias, maxval=bias)
                    decoder_input = miu + tf.exp(sigma) * noise

                    # projection
                    decoder_input = tf.layers.dense(
                        decoder_input,
                        seq_length * reduced_size,
                        activation=util.gelu,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=config.initializer_range),
                        trainable=trainable)
                    intermediate_input = tf.reshape(
                        decoder_input, [-1, seq_length, reduced_size])
                    intermediate_input = util.layer_norm(
                        intermediate_input, trainable=trainable)
                    intermediate_input = util.dropout(
                        intermediate_input, config.hidden_dropout_prob)

                # MLP
                with tf.variable_scope('intermediate'):
                    intermediate_output = tf.layers.dense(
                        intermediate_input,
                        4 * reduced_size,
                        activation=util.gelu,
                        kernel_initializer=util.create_initializer(
                            config.initializer_range),
                        trainable=trainable)
                with tf.variable_scope('output'):
                    decoder_output = tf.layers.dense(
                        intermediate_output,
                        config.hidden_size,
                        kernel_initializer=util.create_initializer(
                            config.initializer_range),
                        trainable=trainable)
                    decoder_output = util.layer_norm(
                        decoder_output, trainable=trainable)
                    decoder_output = util.dropout(
                        decoder_output, config.hidden_dropout_prob)
                self.all_decoder_layers = [intermediate_output, decoder_output]
                self.all_decoder_layers = [decoder_output]

        # reconstruction
        with tf.variable_scope('cls/predictions'):
            with tf.variable_scope('transform'):
                input_tensor = tf.layers.dense(
                    decoder_output,
                    units=config.hidden_size,
                    activation=util.get_activation(config.hidden_act),
                    kernel_initializer=util.create_initializer(
                        config.initializer_range),
                    trainable=trainable)
                input_tensor = util.layer_norm(
                    input_tensor, trainable=trainable)
            output_weights = self.embedding_table
            output_bias = tf.get_variable(
                'output_bias',
                shape=[config.vocab_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)
            flatten_input_tensor = tf.reshape(
                input_tensor, [-1, config.hidden_size])

            logits = tf.matmul(
                flatten_input_tensor, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(
                logits, [batch_size, seq_length, config.vocab_size])
            probs = tf.nn.softmax(logits, axis=-1, name='probs')
            lm_log_probs = tf.nn.log_softmax(logits, axis=-1)

            self._preds['preds'] = tf.argmax(probs, axis=-1)
            one_hot_labels = tf.one_hot(
                input_ids, depth=config.vocab_size, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(
                lm_log_probs * one_hot_labels, axis=[-1])
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            self.total_loss = (
                tf.reduce_mean(per_example_loss) +
                tf.reduce_mean(tf.square(miu)) +
                tf.reduce_mean(tf.exp(sigma) - sigma - 1))
            self._losses['losses'] = per_example_loss


class Config:
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
