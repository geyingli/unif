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
''' ALBERT.
  Code revised from Google's implementation.
  See `https://github.com/google-research/albert`.
'''

import math
import copy
import json
import numpy as np

from uf.tools import tf
from .base import BaseEncoder, BaseDecoder
from . import util


class ALBERTEncoder(BaseEncoder):
  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               batch_size=None,
               seq_length=None,
               use_one_hot_embeddings=False,
               use_einsum=True,
               scope='bert',
               use_tilda_embedding=False,
               drop_pooler=False,
               **kwargs):
    """Constructor for AlbertModel.

    Args:
      config: `AlbertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      use_einsum: (optional) bool. Whether to use einsum or reshape+matmul for
        dense layers
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    if batch_size is None or seq_length is None:
      input_shape = util.get_shape_list(input_ids, expected_rank=2)
      batch_size = input_shape[0]
      seq_length = input_shape[1]

    # Tilda embeddings for SMART algorithm
    tilda_embeddings = None
    if use_tilda_embedding:
      with tf.variable_scope('', reuse=True):
        tilda_embeddings = tf.get_variable('tilda_embeddings')

    with tf.variable_scope(scope):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.output_embedding_table) = embedding_lookup(
             input_ids=input_ids,
             vocab_size=config.vocab_size,
             embedding_size=config.embedding_size,
             initializer_range=config.initializer_range,
             word_embedding_name="word_embeddings",
             use_one_hot_embeddings=use_one_hot_embeddings,
             tilda_embeddings=tilda_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.word_embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings)

      with tf.variable_scope("encoder"):
        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=input_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_hidden_groups=config.num_hidden_groups,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            inner_group_num=config.inner_group_num,
            intermediate_act_fn=util.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            use_einsum=use_einsum)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)

        # trick: ignore the fully connected layer
        if drop_pooler:
          self.pooled_output = first_token_tensor
        else:
          self.pooled_output = tf.layers.dense(
              first_token_tensor,
              config.hidden_size,
              activation=tf.tanh,
              kernel_initializer=util.create_initializer(
                  config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_word_embedding_output(self):
    """Get output of the word(piece) embedding lookup.

    This is BEFORE positional embeddings and token type embeddings have been
    added.

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size]
      corresponding to the output of the word(piece) embedding layer.
    """
    return self.word_embedding_output

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, embedding_size]
      corresponding to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.output_embedding_table



class ALBERTDecoder(BaseDecoder):
    def __init__(self,
                 albert_config,
                 is_training,
                 encoder,
                 masked_lm_positions,
                 masked_lm_ids,
                 masked_lm_weights,
                 sentence_order_labels=None,
                 sample_weight=None,
                 scope_lm='cls/predictions',
                 scope_cls='cls/seq_relationship',
                 name='',
                 trainable=True,
                 **kwargs):
        super(ALBERTDecoder, self).__init__(**kwargs)

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
                    units=albert_config.embedding_size,
                    activation=util.get_activation(albert_config.hidden_act),
                    kernel_initializer=util.create_initializer(
                        albert_config.initializer_range))
                input_tensor = util.layer_norm(input_tensor)
            output_bias = tf.get_variable(
                'output_bias', shape=[albert_config.vocab_size],
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
                label_ids, depth=albert_config.vocab_size, dtype=tf.float32)
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
        if sentence_order_labels is not None:
            with tf.variable_scope(scope_cls):
                output_weights = tf.get_variable(
                    'output_weights',
                    shape=[2, albert_config.hidden_size],
                    initializer=util.create_initializer(
                        albert_config.initializer_range))
                output_bias = tf.get_variable(
                    'output_bias', shape=[2],
                    initializer=tf.zeros_initializer())

                logits = tf.matmul(encoder.get_pooled_output(),
                                   output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                probs = tf.nn.softmax(logits, axis=-1, name='probs')
                log_probs = tf.nn.log_softmax(logits, axis=-1)

                labels = tf.reshape(sentence_order_labels, [-1])
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


def get_timing_signal_1d_given_position(channels,
                                        position,
                                        min_timescale=1.0,
                                        max_timescale=1.0e4):
  """Get sinusoids of diff frequencies, with timing position given.

  Adapted from add_timing_signal_1d_given_position in
  //third_party/py/tensor2tensor/layers/common_attention.py

  Args:
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    position: a Tensor with shape [batch, seq_len]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor of timing signals [batch, seq_len, channels]
  """
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = (
      tf.expand_dims(tf.to_float(position), 2) * tf.expand_dims(
          tf.expand_dims(inv_timescales, 0), 0))
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
  signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
  return signal


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False,
                     tilda_embeddings=None):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  if tilda_embeddings is not None:
    embedding_table = tilda_embeddings
  else:
    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=util.create_initializer(initializer_range))

  if use_one_hot_embeddings:
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.nn.embedding_lookup(embedding_table, input_ids)

  input_shape = util.get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=True):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = util.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=util.create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary, unless converting to tflite model.
    if use_one_hot_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
      token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
    else:
      token_type_embeddings = tf.nn.embedding_lookup(token_type_table,
                                                     token_type_ids)
    output += token_type_embeddings

  if use_position_embeddings:
    full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=util.create_initializer(initializer_range))
    # Since the position embedding table is a learned variable, we create it
    # using a (long) sequence length `max_position_embeddings`. The actual
    # sequence length might be shorter than this, for faster training of
    # tasks that do not have long sequences.
    #
    # So `full_position_embeddings` is effectively an embedding table
    # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    # perform a slice.
    position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                   [seq_length, -1])
    num_dims = len(output.shape.as_list())

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.
    position_broadcast_shape = []
    for _ in range(num_dims - 2):
      position_broadcast_shape.append(1)
    position_broadcast_shape.extend([seq_length, width])
    position_embeddings = tf.reshape(position_embeddings,
                                     position_broadcast_shape)
    output += position_embeddings

  output = util.layer_norm_and_dropout(output, dropout_prob)
  return output


def einsum_via_matmul(input_tensor, w, num_inner_dims):
  """Implements einsum via matmul and reshape ops.

  Args:
    input_tensor: float Tensor of shape [<batch_dims>, <inner_dims>].
    w: float Tensor of shape [<inner_dims>, <outer_dims>].
    num_inner_dims: int. number of dimensions to use for inner products.

  Returns:
    float Tensor of shape [<batch_dims>, <outer_dims>].
  """
  input_shape = util.get_shape_list(input_tensor)
  w_shape = util.get_shape_list(w)
  batch_dims = input_shape[: -num_inner_dims]
  inner_dims = input_shape[-num_inner_dims:]
  outer_dims = w_shape[num_inner_dims:]
  inner_dim = np.prod(inner_dims)
  outer_dim = np.prod(outer_dims)
  if num_inner_dims > 1:
    input_tensor = tf.reshape(input_tensor, batch_dims + [inner_dim])
  if len(w_shape) > 2:
    w = tf.reshape(w, [inner_dim, outer_dim])
  ret = tf.matmul(input_tensor, w)
  if len(outer_dims) > 1:
    ret = tf.reshape(ret, batch_dims + outer_dims)
  return ret


def dense_layer_3d(input_tensor,
                   num_attention_heads,
                   head_size,
                   initializer,
                   activation,
                   use_einsum,
                   name=None):
  """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    num_attention_heads: Number of attention heads.
    head_size: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """

  input_shape = util.get_shape_list(input_tensor)
  hidden_size = input_shape[2]

  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, num_attention_heads * head_size],
        initializer=initializer)
    w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * head_size],
        initializer=tf.zeros_initializer)
    b = tf.reshape(b, [num_attention_heads, head_size])
    if use_einsum:
      ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
    else:
      ret = einsum_via_matmul(input_tensor, w, 1)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_3d_proj(input_tensor,
                        hidden_size,
                        head_size,
                        initializer,
                        activation,
                        use_einsum,
                        name=None):
  """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  input_shape = util.get_shape_list(input_tensor)
  num_attention_heads = input_shape[2]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[num_attention_heads * head_size, hidden_size],
        initializer=initializer)
    w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
    b = tf.get_variable(
        name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)
    if use_einsum:
      ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
    else:
      ret = einsum_via_matmul(input_tensor, w, 2)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dense_layer_2d(input_tensor,
                   output_size,
                   initializer,
                   activation,
                   use_einsum,
                   num_attention_heads=1,
                   name=None):
  """A dense layer with 2D kernel.

  Args:
    input_tensor: Float tensor with rank 3.
    output_size: The size of output dimension.
    initializer: Kernel initializer.
    activation: Activation function.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
    num_attention_heads: number of attention head in attention layer.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
  del num_attention_heads  # unused
  input_shape = util.get_shape_list(input_tensor)
  hidden_size = input_shape[2]
  with tf.variable_scope(name):
    w = tf.get_variable(
        name="kernel",
        shape=[hidden_size, output_size],
        initializer=initializer)
    b = tf.get_variable(
        name="bias", shape=[output_size], initializer=tf.zeros_initializer)
    if use_einsum:
      ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
    else:
      ret = tf.matmul(input_tensor, w)
    ret += b
  if activation is not None:
    return activation(ret)
  else:
    return ret


def dot_product_attention(q, k, v, bias, dropout_rate=0.0):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
  logits = tf.multiply(logits, 1.0 / math.sqrt(float(util.get_shape_list(q)[-1])))
  if bias is not None:
    # `attention_mask` = [B, T]
    from_shape = util.get_shape_list(q)
    if len(from_shape) == 4:
      broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], 1], tf.float32)
    elif len(from_shape) == 5:
      # from_shape = [B, N, Block_num, block_size, depth]#
      broadcast_ones = tf.ones([from_shape[0], 1, from_shape[2], from_shape[3],
                                1], tf.float32)

    bias = tf.matmul(broadcast_ones,
                     tf.cast(bias, tf.float32), transpose_b=True)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - bias) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    logits += adder
  else:
    adder = 0.0

  attention_probs = tf.nn.softmax(logits, name="attention_probs")
  attention_probs = util.dropout(attention_probs, dropout_rate)
  return tf.matmul(attention_probs, v)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_einsum=True):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      The values should be 1 or 0. The attention scores will effectively
      be set to -infinity for any positions in the mask that are 0, and
      will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """
  from_shape = util.get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = util.get_shape_list(to_tensor, expected_rank=[2, 3])
  size_per_head = int(from_shape[2]/num_attention_heads)

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  # `query_layer` = [B, F, N, H]
  q = dense_layer_3d(from_tensor, num_attention_heads, size_per_head,
                     util.create_initializer(initializer_range), query_act,
                     use_einsum, "query")

  # `key_layer` = [B, T, N, H]
  k = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                     util.create_initializer(initializer_range), key_act,
                     use_einsum, "key")
  # `value_layer` = [B, T, N, H]
  v = dense_layer_3d(to_tensor, num_attention_heads, size_per_head,
                     util.create_initializer(initializer_range), value_act,
                     use_einsum, "value")
  q = tf.transpose(q, [0, 2, 1, 3])
  k = tf.transpose(k, [0, 2, 1, 3])
  v = tf.transpose(v, [0, 2, 1, 3])
  if attention_mask is not None:
    attention_mask = tf.reshape(
        attention_mask, [batch_size, 1, to_seq_length, 1])
    # 'new_embeddings = [B, N, F, H]'
  new_embeddings = dot_product_attention(q, k, v, attention_mask,
                                         attention_probs_dropout_prob)

  return tf.transpose(new_embeddings, [0, 2, 1, 3])


def attention_ffn_block(layer_input,
                        hidden_size=768,
                        attention_mask=None,
                        num_attention_heads=1,
                        attention_head_size=64,
                        attention_probs_dropout_prob=0.0,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        initializer_range=0.02,
                        hidden_dropout_prob=0.0,
                        use_einsum=True):
  """A network with attention-ffn as sub-block.

  Args:
    layer_input: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    hidden_size: (optional) int, size of hidden layer.
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    attention_head_size: int. Size of attention head.
    attention_probs_dropout_prob: float. dropout probability for attention_layer
    intermediate_size: int. Size of intermediate hidden layer.
    intermediate_act_fn: (optional) Activation function for the intermediate
      layer.
    initializer_range: float. Range of the weight initializer.
    hidden_dropout_prob: (optional) float. Dropout probability of the hidden
      layer.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers

  Returns:
    layer output
  """

  with tf.variable_scope("attention_1"):
    with tf.variable_scope("self"):
      attention_output = attention_layer(
          from_tensor=layer_input,
          to_tensor=layer_input,
          attention_mask=attention_mask,
          num_attention_heads=num_attention_heads,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          initializer_range=initializer_range,
          use_einsum=use_einsum)

    # Run a linear projection of `hidden_size` then add a residual
    # with `layer_input`.
    with tf.variable_scope("output"):
      attention_output = dense_layer_3d_proj(
          attention_output,
          hidden_size,
          attention_head_size,
          util.create_initializer(initializer_range),
          None,
          use_einsum=use_einsum,
          name="dense")
      attention_output = util.dropout(attention_output, hidden_dropout_prob)
  attention_output = util.layer_norm(attention_output + layer_input)
  with tf.variable_scope("ffn_1"):
    with tf.variable_scope("intermediate"):
      intermediate_output = dense_layer_2d(
          attention_output,
          intermediate_size,
          util.create_initializer(initializer_range),
          intermediate_act_fn,
          use_einsum=use_einsum,
          num_attention_heads=num_attention_heads,
          name="dense")
      with tf.variable_scope("output"):
        ffn_output = dense_layer_2d(
            intermediate_output,
            hidden_size,
            util.create_initializer(initializer_range),
            None,
            use_einsum=use_einsum,
            num_attention_heads=num_attention_heads,
            name="dense")
      ffn_output = util.dropout(ffn_output, hidden_dropout_prob)
  ffn_output = util.layer_norm(ffn_output + attention_output)
  return ffn_output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_hidden_groups=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      inner_group_num=1,
                      intermediate_act_fn="gelu",
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      use_einsum=True):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length],
      with 1 for positions that can be attended to and 0 in positions that
      should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_hidden_groups: int. Number of group for the hidden layers, parameters
      in the same group are shared.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    inner_group_num: int, number of inner repetition of attention and ffn.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = hidden_size // num_attention_heads
  input_shape = util.get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]

  all_layer_outputs = []
  if input_width != hidden_size:
    prev_output = dense_layer_2d(
        input_tensor, hidden_size, util.create_initializer(initializer_range),
        None, use_einsum=use_einsum, name="embedding_hidden_mapping_in")
  else:
    prev_output = input_tensor
  with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
    for layer_idx in range(num_hidden_layers):
      group_idx = int(layer_idx / num_hidden_layers * num_hidden_groups)
      with tf.variable_scope("group_%d" % group_idx):
        with tf.name_scope("layer_%d" % layer_idx):
          layer_output = prev_output
          for inner_group_idx in range(inner_group_num):
            with tf.variable_scope("inner_group_%d" % inner_group_idx):
              layer_output = attention_ffn_block(
                  layer_input=layer_output,
                  hidden_size=hidden_size,
                  attention_mask=attention_mask,
                  num_attention_heads=num_attention_heads,
                  attention_head_size=attention_head_size,
                  attention_probs_dropout_prob=attention_probs_dropout_prob,
                  intermediate_size=intermediate_size,
                  intermediate_act_fn=intermediate_act_fn,
                  initializer_range=initializer_range,
                  hidden_dropout_prob=hidden_dropout_prob,
                  use_einsum=use_einsum)
              prev_output = layer_output
              all_layer_outputs.append(layer_output)
  if do_return_all_layers:
    return all_layer_outputs
  else:
    return all_layer_outputs[-1]


class ALBERTConfig:
  """Configuration for `AlbertModel`.

  The default settings match the configuration of model `albert_xxlarge`.
  """

  def __init__(self,
               vocab_size,
               embedding_size=128,
               hidden_size=4096,
               num_hidden_layers=12,
               num_hidden_groups=1,
               num_attention_heads=64,
               intermediate_size=16384,
               inner_group_num=1,
               down_scale_factor=1,
               hidden_act="gelu",
               hidden_dropout_prob=0,
               attention_probs_dropout_prob=0,
               max_position_embeddings=512,
               type_vocab_size=2,
               initializer_range=0.02):
    """Constructs AlbertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `AlbertModel`.
      embedding_size: size of voc embeddings.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_hidden_groups: Number of group for the hidden layers, parameters in
        the same group are shared.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      inner_group_num: int, number of inner repetition of attention and ffn.
      down_scale_factor: float, the scale to apply
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `AlbertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_hidden_groups = num_hidden_groups
    self.num_attention_heads = num_attention_heads
    self.inner_group_num = inner_group_num
    self.down_scale_factor = down_scale_factor
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `AlbertConfig` from a Python dictionary of parameters."""
    config = cls(vocab_size=None)
    for (key, value) in json_object.items():
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `AlbertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
