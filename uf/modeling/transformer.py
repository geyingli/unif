# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Transformer.
  Code revised from Kyubyong's implementation.
  See `https://github.com/Kyubyong/transformer`.
'''

import numpy as np

from uf.tools import tf, contrib
from .base import BaseEncoder, BaseDecoder
from . import util


class Transformer(BaseDecoder, BaseEncoder):
    def __init__(self,
                 vocab_size,
                 is_training,
                 source_ids,
                 target_ids,
                 sos_id,
                 sample_weight=None,
                 hidden_size=768,
                 num_blocks=6,
                 num_attention_heads=12,
                 scope='transformer',
                 use_label_smoothing=False,
                 use_tilda_embedding=False,
                 trainable=True,
                 **kwargs):
        super().__init__()

        dropout_rate = 0.0
        if is_training:
            dropout_rate = 0.1

        source_shape = util.get_shape_list(source_ids, expected_rank=2)
        target_shape = util.get_shape_list(target_ids, expected_rank=2)
        batch_size = source_shape[0]
        source_max_seq_length = source_shape[1]
        target_max_seq_length = target_shape[1]

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):
            source_mask = tf.math.equal(source_ids, 0)

            # embedding
            with tf.variable_scope('embeddings'):
                (enc, embedding_table) = embedding_lookup(
                    input_ids=source_ids,
                    vocab_size=vocab_size,
                    batch_size=batch_size,
                    max_seq_length=source_max_seq_length,
                    embedding_size=hidden_size,
                    word_embedding_name='word_embeddings',
                    tilda_embeddings=tilda_embeddings)
                enc *= hidden_size ** 0.5  # scale
                enc += positional_encoding(enc, source_max_seq_length)
                enc = util.dropout(enc, dropout_rate)

            with tf.variable_scope('encoder'):

                # stacked multi-attention layers
                for i in range(num_blocks):
                    with tf.variable_scope('block_%s' % i):

                        # self-attention
                        enc = multihead_attention(
                            queries=enc,
                            keys=enc,
                            values=enc,
                            key_masks=source_mask,
                            num_heads=num_attention_heads,
                            dropout_rate=dropout_rate,
                            training=is_training,
                            causality=False,
                            scope='self_attention')

                        # feed forward
                        enc = ff(enc, num_units=[hidden_size * 4, hidden_size])
                memory = enc

            def _forward(target_ids, target_mask, target_max_seq_length):

                with tf.variable_scope('decoder'):

                    # shared embedding
                    dec = tf.nn.embedding_lookup(embedding_table, target_ids)
                    dec *= hidden_size ** 0.5  # scale
                    dec += positional_encoding(dec, target_max_seq_length)
                    dec = util.dropout(dec, dropout_rate)

                    # blocks
                    for i in range(num_blocks):
                        with tf.variable_scope('block_%s' % i):

                            # masked self-attention
                            dec = multihead_attention(
                                queries=dec,
                                keys=dec,
                                values=dec,
                                key_masks=target_mask,
                                num_heads=num_attention_heads,
                                dropout_rate=dropout_rate,
                                training=is_training,
                                causality=True,
                                scope='masked_self_attention')

                            # vanilla attention
                            dec = multihead_attention(
                                queries=dec,
                                keys=memory,
                                values=memory,
                                key_masks=source_mask,
                                num_heads=num_attention_heads,
                                dropout_rate=dropout_rate,
                                training=is_training,
                                causality=False,
                                scope='vanilla_attention')

                            # feed forward
                            dec = ff(
                                dec, num_units=[4 * hidden_size, hidden_size])

                # final linear projection (embedding weights are shared)
                with tf.variable_scope('cls'):
                    output_bias = tf.get_variable(
                        'output_bias', shape=[vocab_size],
                        initializer=tf.zeros_initializer())
                    dec = tf.reshape(dec, [-1, hidden_size])
                    logits = tf.matmul(dec, embedding_table, transpose_b=True)
                    logits = tf.reshape(
                        logits, [-1, target_max_seq_length, vocab_size])
                    logits = tf.nn.bias_add(logits, output_bias)

                return logits

            # convert to labels
            label_ids = tf.concat(
                [target_ids[:, 1:],
                 tf.zeros([batch_size, 1], dtype=tf.int32)], axis=-1)

            # forward once
            if is_training:
                target_mask = tf.math.equal(target_ids, 0)  # (N, T2)
                logits = _forward(
                    target_ids, target_mask, target_max_seq_length)

                self.preds['MT'] = tf.argmax(logits, axis=-1)

            # forward loop
            else:
                target_mask_base = tf.zeros([batch_size, 1], dtype=tf.int32)
                target_ids = tf.ones([batch_size, 1], dtype=tf.int32) * sos_id

                for cur_length in range(1, target_max_seq_length + 1):
                    target_mask = tf.tile(target_mask_base, [1, cur_length])
                    logits = _forward(target_ids, target_mask, cur_length)

                    pred_ids = tf.argmax(
                        logits[:, cur_length-1:cur_length, :],
                        axis=-1)
                    pred_ids = tf.cast(pred_ids, tf.int32)
                    target_ids = tf.concat([target_ids, pred_ids], axis=-1)

                self.preds['MT'] = target_ids[:, 1:]

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids, depth=vocab_size)
            if use_label_smoothing:
                one_hot_labels = label_smoothing(one_hot_labels)
            per_token_loss = -tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)
            label_mask = tf.cast(tf.not_equal(label_ids, 0), tf.float32)
            per_example_loss = \
                tf.reduce_sum(per_token_loss * label_mask, axis=-1) / \
                tf.reduce_sum(label_mask, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            self.total_loss = tf.reduce_mean(per_example_loss)
            self.losses['MT'] = per_example_loss


def embedding_lookup(input_ids,
                     vocab_size,
                     batch_size,
                     max_seq_length,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name='word_embeddings',
                     zero_pad=True,
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

    embedding_table = tf.concat(
        (tf.zeros(shape=[1, embedding_size]),
         embedding_table[1:, :]), axis=0)

    flat_input_ids = tf.reshape(input_ids, [-1])
    output = tf.gather(
        embedding_table, flat_input_ids, name='embedding_look_up')
    output = tf.reshape(
        output, [batch_size, max_seq_length, embedding_size])

    return (output, embedding_table)


def ln(inputs, epsilon = 1e-8, scope='ln'):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope('shared_weight_matrix'):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=\
                                       contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def scaled_dot_product_attention(Q, K, V, key_masks,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope='scaled_dot_product_attention'):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, key_masks=key_masks, type='key')

        # causality or future blinding masking
        if causality:
            outputs = mask(outputs, type='future')

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image('attention', tf.expand_dims(attention[:1], -1))

        # # query masking
        # outputs = mask(outputs, Q, K, type='query')

        # dropout
        outputs = tf.layers.dropout(
            outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs


def mask(inputs, key_masks=None, type=None):
    '''Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. 'key' | 'future'

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type='key')
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    '''
    padding_num = -2 ** 32 + 1
    if type in ('k', 'key', 'keys'):
        key_masks = tf.to_float(key_masks)
        key_masks = tf.tile(
            key_masks,
            [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
        key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    # elif type in ('q', 'query', 'queries'):
    #     # Generate masks
    #     masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    #     masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    #     masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)
    #
    #     # Apply masks to inputs
    #     outputs = inputs*masks
    elif type in ('f', 'future', 'right'):
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.linalg.LinearOperatorLowerTriangular(
            diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.tile(
            tf.expand_dims(tril, 0),
            [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
    else:
        print('Check if you entered type correctly!')

    return outputs


def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope='multihead_attention'):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        Q = tf.layers.dense(
            queries, d_model, use_bias=True) # (N, T_q, d_model)
        K = tf.layers.dense(
            keys, d_model, use_bias=True) # (N, T_k, d_model)
        V = tf.layers.dense(
            values, d_model, use_bias=True) # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(
            tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(
            tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(
            tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(
            Q_, K_, V_, key_masks, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(
            tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs

def ff(inputs, num_units, scope='positionwise_feedforward'):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs

def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope='positional_encoding'):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(
            position_enc, tf.float32) # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)
