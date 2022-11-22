""" ALBERT.
  Code revised from Google's implementation.
  See `https://github.com/google-research/albert`.
"""

import math
import copy
import json
import random
import collections
import numpy as np

from ...third import tf
from .._base_._base_ import BaseEncoder, BaseDecoder
from .. import util


class ALBERTEncoder(BaseEncoder):
  def __init__(self,
               albert_config,
               is_training,
               input_ids,
               input_mask=None,
               segment_ids=None,
               scope="bert",
               drop_pooler=False,
               trainable=True,
               **kwargs):
    """Constructor for AlbertModel.

    Args:
      albert_config: `AlbertConfig` instance.
      is_training: bool. true for training model, false for eval model.
        Controls whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      segment_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_einsum: (optional) bool. Whether to use einsum or reshape+matmul for
        dense layers
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """

    albert_config = copy.deepcopy(albert_config)
    if not is_training:
      albert_config.hidden_dropout_prob = 0.0
      albert_config.attention_probs_dropout_prob = 0.0

    input_shape = util.get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if segment_ids is None:
      segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.word_embedding_output, self.output_embedding_table) = embedding_lookup(
             input_ids=input_ids,
             vocab_size=albert_config.vocab_size,
             embedding_size=albert_config.embedding_size,
             initializer_range=albert_config.initializer_range,
             word_embedding_name="word_embeddings",
             tilda_embeddings=kwargs.get("tilda_embeddings"),
             trainable=trainable)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.word_embedding_output,
            use_token_type=True,
            segment_ids=segment_ids,
            token_type_vocab_size=albert_config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=albert_config.initializer_range,
            max_position_embeddings=albert_config.max_position_embeddings,
            dropout_prob=albert_config.hidden_dropout_prob,
            trainable=trainable)

      with tf.variable_scope("encoder"):
        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=input_mask,
            hidden_size=albert_config.hidden_size,
            num_hidden_layers=albert_config.num_hidden_layers,
            num_hidden_groups=albert_config.num_hidden_groups,
            num_attention_heads=albert_config.num_attention_heads,
            intermediate_size=albert_config.intermediate_size,
            inner_group_num=albert_config.inner_group_num,
            intermediate_act_fn=util.get_activation(albert_config.hidden_act),
            hidden_dropout_prob=albert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=albert_config.attention_probs_dropout_prob,
            initializer_range=albert_config.initializer_range,
            do_return_all_layers=True,
            use_einsum=False,
            trainable=trainable)

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
              albert_config.hidden_size,
              activation=tf.tanh,
              kernel_initializer=util.create_initializer(
                  albert_config.initializer_range),
              trainable=trainable)

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
                 scope_lm="cls/predictions",
                 scope_cls="cls/seq_relationship",
                 trainable=True,
                 **kwargs):
        super(ALBERTDecoder, self).__init__(**kwargs)

        if kwargs.get("return_hidden"):
            self.tensors["hidden"] = encoder.get_pooled_output()

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
            with tf.variable_scope("transform"):
                input_tensor = tf.layers.dense(
                    input_tensor,
                    units=albert_config.embedding_size,
                    activation=util.get_activation(albert_config.hidden_act),
                    kernel_initializer=util.create_initializer(
                        albert_config.initializer_range),
                    trainable=trainable)
                input_tensor = util.layer_norm(input_tensor)
            output_bias = tf.get_variable(
                "output_bias", shape=[albert_config.vocab_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            logits = tf.matmul(input_tensor, encoder.get_embedding_table(), transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probs = tf.nn.softmax(logits, axis=-1, name="MLM_probs")
            probs = tf.reshape(probs, [-1, masked_lm_positions.shape[-1], albert_config.vocab_size])

            label_ids = tf.reshape(masked_lm_ids, [-1])
            if sample_weight is not None:
                sample_weight = tf.expand_dims(sample_weight, axis=-1)
                masked_lm_weights *= sample_weight
            label_weights = tf.reshape(masked_lm_weights, [-1])
            per_example_loss = util.cross_entropy(logits, label_ids, albert_config.vocab_size, **kwargs)
            per_example_loss *= tf.reshape(masked_lm_weights, [-1])

            numerator = tf.reduce_sum(per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-5
            scalar_loss = numerator / denominator

            scalar_losses.append(scalar_loss)
            self.tensors["MLM_losses"] = per_example_loss
            self.tensors["MLM_preds"] = tf.argmax(probs, axis=-1)

        # next sentence prediction
        if sentence_order_labels is not None:
            with tf.variable_scope(scope_cls):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[2, albert_config.hidden_size],
                    initializer=util.create_initializer(
                        albert_config.initializer_range),
                    trainable=trainable)
                output_bias = tf.get_variable(
                    "output_bias", shape=[2],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable)

                logits = tf.matmul(encoder.get_pooled_output(),
                                   output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                probs = tf.nn.softmax(logits, axis=-1, name="probs")

                label_ids = tf.reshape(sentence_order_labels, [-1])
                per_example_loss = util.cross_entropy(logits, label_ids, 2, **kwargs)
                if sample_weight is not None:
                    per_example_loss *= sample_weight
                scalar_loss = tf.reduce_mean(per_example_loss)

                scalar_losses.append(scalar_loss)
                self.tensors["SOP_losses"] = per_example_loss
                self.tensors["SOP_probs"] = probs
                self.tensors["SOP_preds"] = tf.argmax(probs, axis=-1)

        self.train_loss = tf.add_n(scalar_losses)


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
                     tilda_embeddings=None,
                     trainable=True):
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
        initializer=util.create_initializer(initializer_range),
        trainable=trainable)

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
                            segment_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=True,
                            trainable=True):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `segment_ids`.
    segment_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `segment_ids`.
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
    if segment_ids is None:
      raise ValueError("`segment_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=util.create_initializer(initializer_range),
        trainable=trainable)
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary, unless converting to tflite model.
    if use_one_hot_embeddings:
      flat_token_type_ids = tf.reshape(segment_ids, [-1])
      one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
      token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
    else:
      token_type_embeddings = tf.nn.embedding_lookup(token_type_table,
                                                     segment_ids)
    output += token_type_embeddings

  if use_position_embeddings:
    full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=util.create_initializer(initializer_range),
          trainable=trainable)
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
                   name=None,
                   trainable=True):
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
        initializer=initializer,
        trainable=trainable)
    w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
    b = tf.get_variable(
        name="bias",
        shape=[num_attention_heads * head_size],
        initializer=tf.zeros_initializer,
        trainable=trainable)
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
                        name=None,
                        trainable=True):
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
        initializer=initializer, trainable=trainable)
    w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
    b = tf.get_variable(
        name="bias", shape=[hidden_size],
        initializer=tf.zeros_initializer, trainable=trainable)
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
                   name=None,
                   trainable=True):
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
        initializer=initializer, trainable=trainable)
    b = tf.get_variable(
        name="bias", shape=[output_size],
        initializer=tf.zeros_initializer, trainable=trainable)
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
    # "new_embeddings = [B, N, F, H]"
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
                      use_einsum=True,
                      trainable=True):
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
      type_vocab_size: The vocabulary size of the `segment_ids` passed into
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


def create_instances_from_document(
    all_documents, document_index, max_seq_length, masked_lm_prob,
    max_predictions_per_seq, short_seq_prob, vocab_words, random_next_sentence=False,
):
    document = all_documents[document_index]

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_seq_length
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_seq_length)

    # We DON"T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into
                # the `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or (random_next_sentence and random.random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for
                    # large corpora. However, just to be careful, we try to
                    # make sure that the random document is not the same as
                    # the document we"re processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them
                    # back" so they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not random_next_sentence and random.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    tokens_a, tokens_b = tokens_b, tokens_a
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instances.append(([tokens_a, tokens_b], is_random_next))
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words,
    ngram=3, favor_shorterngram=True, do_permutation=False, do_whole_word_mask=True,
):
    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not favor_shorterngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or
        # previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(
            ngrams[:len(cand_index_set)],
            p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True),
        )
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    random.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or
            # previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] /
                pvals[:len(cand_index_set)].sum(keepdims=True),
            )
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        random.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": 4,
        "/embedding_hidden_mapping_in": 3,
        "/group_0": 2,
        "/pooler/": 1,
        "cls/": 0,
    }
    return decay_power
