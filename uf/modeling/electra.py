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
''' ELECTRA.
  Code revised from Google's implementation.
  See `https://github.com/google-research/electra`.
'''

import math
import copy
import collections

from uf.tools import tf
from .base import BaseDecoder
from .bert import BERTConfig
from . import util


Inputs = collections.namedtuple(
    'Inputs', ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
               'masked_lm_ids', 'masked_lm_weights'])


class ELECTRA(BaseDecoder):
    def __init__(self,
                 model_size,
                 is_training,
                 placeholders,
                 sample_weight,
                 vocab,
                 electra_objective=False,
                 **kwargs):
        super().__init__()

        # set up configs
        self.config = PretrainingConfig(model_size=model_size, **kwargs)
        self.bert_config = get_bert_config(
            model_size, len(list(vocab.keys())))

        # alert
        if is_training:
            if electra_objective:
                tf.logging.info(
                    'Training on Generator and Discriminator together.')
            else:
                tf.logging.info(
                    'Training on Generator, with Discriminator frozen. '
                    '(Pass `electra_objective=True` to include '
                    'Discriminator)')

        # Generator
        masked_inputs = features_to_inputs(placeholders)
        embedding_size = (self.bert_config.hidden_size
                          if self.config.embedding_size is None
                          else self.config.embedding_size)
        generator = self._build_transformer(
            masked_inputs, is_training,
            bert_config=get_generator_config(self.config, self.bert_config),
            embedding_size=embedding_size,
            untied_embeddings=False,
            name='generator')
        mlm_output = self._get_generator_output(
            masked_inputs, sample_weight, generator)
        self.total_loss = self.config.gen_weight * mlm_output.loss
        self.losses['MLM'] = mlm_output.per_example_loss
        self.preds['MLM'] = mlm_output.preds

        # Discriminator
        fake_data = self._get_fake_data(masked_inputs, mlm_output.logits)
        if is_training:
            disc_input = fake_data.inputs
        else:
            disc_input = masked_inputs
        discriminator = self._build_transformer(
            disc_input, is_training,
            bert_config=self.bert_config,
            reuse=False,
            embedding_size=embedding_size,
            name='electra')
        disc_output = self._get_discriminator_output(
            disc_input, sample_weight, discriminator,
            fake_data.is_fake_tokens)
        if electra_objective:
            self.total_loss += self.config.disc_weight * disc_output.loss
        self.losses['RTD'] = disc_output.per_example_loss
        self.probs['RTD'] = disc_output.probs
        self.preds['RTD'] = disc_output.preds
        self.preds['RTD_labels'] = disc_output.labels

    def _get_generator_output(self, inputs, sample_weight, generator):
        '''Masked language modeling softmax layer.'''

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

        input_tensor = gather_indexes(
            generator.get_sequence_output(), inputs.masked_lm_positions)
        with tf.variable_scope('generator_predictions'):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=self.config.embedding_size,
                activation=util.get_activation(self.bert_config.hidden_act),
                kernel_initializer=util.create_initializer(
                    self.bert_config.initializer_range))
            input_tensor = util.layer_norm(input_tensor)
            output_bias = tf.get_variable(
                'output_bias', shape=[self.bert_config.vocab_size],
                initializer=tf.zeros_initializer())

            logits = tf.matmul(
                input_tensor, generator.get_embedding_table(),
                transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probs = tf.nn.softmax(logits, axis=-1, name='MLM_probs')
            preds = tf.argmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            label_ids = tf.reshape(inputs.masked_lm_ids, [-1])
            masked_lm_weights = inputs.masked_lm_weights
            if sample_weight is not None:
                sample_weight = tf.expand_dims(
                    tf.cast(sample_weight, dtype=tf.float32), axis=-1)
                masked_lm_weights *= sample_weight
            label_weights = tf.reshape(masked_lm_weights, [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=self.bert_config.vocab_size, dtype=tf.float32)
            per_example_loss = - tf.reduce_sum(
                log_probs * one_hot_labels, axis=[-1])
            per_example_loss = label_weights * per_example_loss

            numerator = tf.reduce_sum(per_example_loss)
            denominator = tf.reduce_sum(label_weights) + 1e-6
            loss = numerator / denominator

            MLMOutput = collections.namedtuple(
                'MLMOutput',
                ['logits', 'probs', 'loss', 'per_example_loss', 'preds'])
            return MLMOutput(
                logits=logits,
                probs=probs,
                per_example_loss=per_example_loss,
                loss=loss,
                preds=preds)

    def _get_discriminator_output(self, inputs, sample_weight, discriminator,
                                  labels):
        '''Discriminator binary classifier.'''
        with tf.variable_scope('discriminator_predictions'):
            hidden = tf.layers.dense(
                discriminator.get_sequence_output(),
                units=self.bert_config.hidden_size,
                activation=util.get_activation(self.bert_config.hidden_act),
                kernel_initializer=util.create_initializer(
                    self.bert_config.initializer_range))
            logits = tf.squeeze(tf.layers.dense(hidden, units=1), -1)
            weights = tf.cast(inputs.input_mask, tf.float32)
            labelsf = tf.cast(labels, tf.float32)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits, labels=labelsf) * weights
            per_example_loss = (tf.reduce_sum(losses, axis=-1) /
                                (1e-6 + tf.reduce_sum(weights, axis=-1)))
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, dtype=tf.float32)
                per_example_loss *= sample_weight
            loss = tf.reduce_sum(losses) / (1e-6 + tf.reduce_sum(weights))
            probs = tf.nn.sigmoid(logits)
            preds = tf.cast(tf.greater(probs, 0.5), tf.int32)
            DiscOutput = collections.namedtuple(
                'DiscOutput', ['loss', 'per_example_loss', 'probs', 'preds',
                               'labels'])
            return DiscOutput(
                loss=loss, per_example_loss=per_example_loss, probs=probs,
                preds=preds, labels=labels)

    def _get_fake_data(self, inputs, mlm_logits):
        '''Sample from the generator to create corrupted input.'''
        inputs = unmask(inputs)
        disallow = tf.one_hot(
            inputs.masked_lm_ids, depth=self.bert_config.vocab_size,
            dtype=tf.float32) if self.config.disallow_correct else None
        sampled_tokens = tf.stop_gradient(sample_from_softmax(
            mlm_logits / self.config.temperature, disallow=disallow))
        sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
        updated_input_ids, masked = scatter_update(
            inputs.input_ids, sampled_tokids, inputs.masked_lm_positions)
        labels = masked * (1 - tf.cast(
            tf.equal(updated_input_ids, inputs.input_ids), tf.int32))
        updated_inputs = get_updated_inputs(
            inputs, input_ids=updated_input_ids)
        FakedData = collections.namedtuple('FakedData', [
            'inputs', 'is_fake_tokens', 'sampled_tokens'])
        return FakedData(inputs=updated_inputs, is_fake_tokens=labels,
                         sampled_tokens=sampled_tokens)

    def _build_transformer(self, inputs, is_training, bert_config,
                           name='electra', reuse=False, **kwargs):
        '''Build a transformer encoder network.'''
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            return BERTEncoder(
                bert_config=bert_config,
                is_training=is_training,
                input_ids=inputs.input_ids,
                input_mask=inputs.input_mask,
                token_type_ids=inputs.segment_ids,
                use_one_hot_embeddings=False,
                scope=name,
                **kwargs)


class BERTEncoder:
    '''BERT model. Although the training algorithm is different, the transformer
    model for ELECTRA is the same as BERT's.

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
      input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    '''

    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None,
                 embedding_size=None,
                 input_embeddings=None,
                 input_reprs=None,
                 update_embeddings=True,
                 untied_embeddings=False):
        '''Constructor for BertModel.

        Args:
          bert_config: `BertConfig` instance.
          is_training: bool. true for training model, false for eval model.
            Controls whether dropout will be applied.
          input_ids: int32 Tensor of shape [batch_size, seq_length].
          input_mask: (optional) int32 Tensor of shape [batch_size,
            seq_length].
          token_type_ids: (optional) int32 Tensor of shape [batch_size,
            seq_length].
          use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
            embeddings or tf.embedding_lookup() for the word embeddings. On
            the TPU, it is much faster if this is True, on the CPU or GPU,
            it is faster if this is False.
          scope: (optional) variable scope. Defaults to 'electra'.

        Raises:
          ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        '''
        bert_config = copy.deepcopy(bert_config)
        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(token_type_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        assert token_type_ids is not None

        if input_reprs is None:
            with tf.variable_scope(
                    ((scope if untied_embeddings else 'electra') +
                     '/embeddings'),
                    reuse=tf.AUTO_REUSE):
                # Perform embedding lookup on the word ids
                if embedding_size is None:
                    embedding_size = bert_config.hidden_size
                (token_embeddings, self.embedding_table) = \
                    embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=bert_config.vocab_size,
                        embedding_size=embedding_size,
                        initializer_range=bert_config.initializer_range,
                        word_embedding_name='word_embeddings',
                        use_one_hot_embeddings=use_one_hot_embeddings)

            with tf.variable_scope(
                    ((scope if untied_embeddings else 'electra') +
                     '/embeddings'), reuse=tf.AUTO_REUSE):
                # Add positional embeddings and token type embeddings, then
                # layer normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=token_embeddings,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=bert_config.initializer_range,
                    max_position_embeddings=\
                        bert_config.max_position_embeddings,
                    dropout_prob=bert_config.hidden_dropout_prob)
        else:
            self.embedding_output = input_reprs
        if not update_embeddings:
            self.embedding_output = tf.stop_gradient(self.embedding_output)

        with tf.variable_scope(scope, default_name='electra'):
            if self.embedding_output.shape[-1] != bert_config.hidden_size:
                self.embedding_output = tf.layers.dense(
                    self.embedding_output, bert_config.hidden_size,
                    name='embeddings_project')

            with tf.variable_scope('encoder'):
                # This converts a 2D mask of shape [batch_size, seq_length]
                # to a 3D mask of shape [batch_size, seq_length, seq_length]
                # which is used for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                    token_type_ids, input_mask)

                # Run the stacked transformer. Output shapes
                # attn_maps:
                #   [n_layers, batch_size, n_heads, seq_length, seq_length]
                (self.all_layer_outputs, self.attn_maps) = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=bert_config.hidden_size,
                    num_hidden_layers=bert_config.num_hidden_layers,
                    num_attention_heads=bert_config.num_attention_heads,
                    intermediate_size=bert_config.intermediate_size,
                    intermediate_act_fn=util.get_activation(
                        bert_config.hidden_act),
                    hidden_dropout_prob=bert_config.hidden_dropout_prob,
                    attention_probs_dropout_prob=
                    bert_config.attention_probs_dropout_prob,
                    initializer_range=bert_config.initializer_range,
                    do_return_all_layers=True)
                self.sequence_output = self.all_layer_outputs[-1]
                self.pooled_output = self.sequence_output[:, 0]

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        '''Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size]
          corresponding to the final hidden of the transformer encoder.
        '''
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_layer_outputs

    def get_embedding_output(self):
        '''Gets output of the embedding lookup (i.e., input to the
        transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size]
          corresponding to the output of the embedding layer, after summing
          the word embeddings with the positional embeddings and the token
          type embeddings, then performing layer normalization. This is the
          input to the transformer.
        '''
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name='word_embeddings',
                     use_one_hot_embeddings=False):
  '''Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.nn.embedding_lookup()`. One hot is better
      for TPUs.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  '''
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  original_dims = input_ids.shape.ndims
  if original_dims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=util.create_initializer(initializer_range))

  if original_dims == 3:
    input_shape = util.get_shape_list(input_ids)
    tf.reshape(input_ids, [-1, input_shape[-1]])
    output = tf.matmul(input_ids, embedding_table)
    output = tf.reshape(output,
                        [input_shape[0], input_shape[1], embedding_size])
  else:
    if use_one_hot_embeddings:
      flat_input_ids = tf.reshape(input_ids, [-1])
      one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
      output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
      output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = util.get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return output, embedding_table


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name='token_type_embeddings',
                            use_position_embeddings=True,
                            position_embedding_name='position_embeddings',
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  '''Performs various post-processing on a word embedding tensor.

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
    dropout_prob: float. Dropout probability applied to the final output
      tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  '''
  input_shape = util.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError('`token_type_ids` must be specified if'
                       '`use_token_type` is True.')
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=util.create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
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

      # Only the last two dimensions are relevant (`seq_length` and `width`),
      # so we broadcast among the first dimensions, which is typically just
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


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  '''Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  '''
  from_shape = util.get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = util.get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def attention_layer(from_tensor,
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
                    from_seq_length=None,
                    to_seq_length=None):
  '''Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on 'Attention
  is all you Need'. If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a 'query' tensor and
  `to_tensor` into 'key' and 'value' tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions
      in the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch
      size of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq
      length of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  '''

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = util.get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = util.get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        'The rank of `from_tensor` must match the rank of `to_tensor`.')

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if batch_size is None or from_seq_length is None or to_seq_length is None:
      raise ValueError(
          'When passing in rank 2 tensors to attention_layer, the values '
          'for `batch_size`, `from_seq_length`, and `to_seq_length` '
          'must all be specified.')

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = util.reshape_to_matrix(from_tensor)
  to_tensor_2d = util.reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name='query',
      kernel_initializer=util.create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name='key',
      kernel_initializer=util.create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name='value',
      kernel_initializer=util.create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between 'query' and 'key' to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = util.dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer, attention_probs


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=util.gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  '''Multi-headed, multi-layer Transformer from 'Attention is All You Need'.

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/
    tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the 'intermediate' (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  '''
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        'The hidden size (%d) is not a multiple of the number of attention '
        'heads (%d)' % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = util.get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError('The width of the input tensor (%d) != hidden size (%d)'
                     % (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = util.reshape_to_matrix(input_tensor)

  attn_maps = []
  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope('layer_%d' % layer_idx):
      with tf.variable_scope('attention'):
        attention_heads = []
        with tf.variable_scope('self'):
          attention_head, probs = attention_layer(
              from_tensor=prev_output,
              to_tensor=prev_output,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)
          attn_maps.append(probs)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope('output'):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=util.create_initializer(initializer_range))
          attention_output = util.dropout(
              attention_output, hidden_dropout_prob)
          attention_output = util.layer_norm(attention_output + prev_output)

      # The activation is only applied to the 'intermediate' hidden layer.
      with tf.variable_scope('intermediate'):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=util.create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope('output'):
        prev_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=util.create_initializer(initializer_range))
        prev_output = util.dropout(prev_output, hidden_dropout_prob)
        prev_output = util.layer_norm(prev_output + attention_output)
        all_layer_outputs.append(prev_output)

  attn_maps = tf.stack(attn_maps, 0)
  if do_return_all_layers:
    return tf.stack([util.reshape_from_matrix(layer, input_shape)
                     for layer in all_layer_outputs], 0), attn_maps
  else:
    return util.reshape_from_matrix(prev_output, input_shape), attn_maps


class PretrainingConfig:
    '''Defines pre-training hyperparameters.'''

    def __init__(self, **kwargs):

        # loss functions
        self.electra_objective = True
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss

        # generator settings
        self.generator_layers = 1.0  # frac of discriminator layers
                                     # for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size
                                           # for gen
        self.disallow_correct = False  # force the generator to sample
                                       # incorrect tokens (so 15% of tokens
                                       # are always fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # update defaults with passed-in hyperparameters
        self.update(kwargs)

        # model settings
        if self.model_size == 'small':
            self.embedding_size = 128
        elif self.model_size == 'base':
            self.embedding_size = 768
        else:
            self.embedding_size = 1024

    def update(self, kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v


def features_to_inputs(features):
  return Inputs(
      input_ids=features['input_ids'],
      input_mask=features['input_mask'],
      segment_ids=features['segment_ids'],
      masked_lm_positions=features.get('masked_lm_positions'),
      masked_lm_ids=features.get('masked_lm_ids'),
      masked_lm_weights=features.get('masked_lm_weights'))


def unmask(inputs):
    unmasked_input_ids, _ = scatter_update(
        inputs.input_ids, inputs.masked_lm_ids, inputs.masked_lm_positions)
    return get_updated_inputs(inputs, input_ids=unmasked_input_ids)


def gather_positions(sequence, positions):
    '''Gathers the vectors at the specific positions over a minibatch.

    Args:
      sequence: A [batch_size, seq_length] or
          [batch_size, seq_length, depth] tensor of values
      positions: A [batch_size, n_positions] tensor of indices

    Returns: A [batch_size, n_positions] or
      [batch_size, n_positions, depth] tensor of the values at the indices
    '''
    shape = util.get_shape_list(sequence, expected_rank=[2, 3])
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
    else:
        B, L = shape
        D = 1
        sequence = tf.expand_dims(sequence, -1)
    position_shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(positions + position_shift, [-1])
    flat_sequence = tf.reshape(sequence, [B * L, D])
    gathered = tf.gather(flat_sequence, flat_positions)
    if depth_dimension:
        return tf.reshape(gathered, [B, -1, D])
    else:
        return tf.reshape(gathered, [B, -1])


def sample_from_softmax(logits, disallow=None):
    if disallow is not None:
        logits -= 1000.0 * tf.reshape(disallow, [-1, logits.shape[-1]])
    uniform_noise = tf.random_uniform(
        util.get_shape_list(logits), minval=0, maxval=1)
    gumbel_noise = -tf.log(-tf.log(uniform_noise + 1e-9) + 1e-9)
    return tf.one_hot(tf.argmax(logits + gumbel_noise, -1,
                                output_type=tf.int32), logits.shape[-1])


def get_updated_inputs(inputs, **kwargs):
    features = inputs._asdict()
    for k, v in kwargs.items():
        features[k] = v
    return features_to_inputs(features)


def scatter_update(sequence, updates, positions):
    '''Scatter-update a sequence.

    Args:
      sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
      updates: A tensor of size batch_size*seq_len(*depth)
      positions: A [batch_size, n_positions] tensor

    Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
      [batch_size, seq_len, depth] tensor of 'sequence' with elements at
      'positions' replaced by the values at 'updates.' Updates to index 0 are
      ignored. If there are duplicated positions the update is only applied
      once. Second is a [batch_size, seq_len] mask tensor of which inputs were
      updated.
    '''
    shape = util.get_shape_list(sequence, expected_rank=[2, 3])
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
        B, L, D = shape
    else:
        B, L = shape
        D = 1
        sequence = tf.expand_dims(sequence, -1)
    N = util.get_shape_list(positions)[1]

    shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(positions + shift, [-1, 1])
    flat_updates = tf.reshape(updates, [-1, D])
    updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
    updates = tf.reshape(updates, [B, L, D])

    flat_updates_mask = tf.ones([B * N], tf.int32)
    updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
    updates_mask = tf.reshape(updates_mask, [B, L])
    not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                                 tf.ones((B, L - 1), tf.int32)], -1)
    updates_mask *= not_first_token
    updates_mask_3d = tf.expand_dims(updates_mask, -1)

    # account for duplicate positions
    if sequence.dtype == tf.float32:
        updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
        updates /= tf.maximum(1.0, updates_mask_3d)
    else:
        assert sequence.dtype == tf.int32
        updates = tf.divide(updates, tf.maximum(1, updates_mask_3d))
        updates = tf.cast(updates, tf.int32)
    updates_mask = tf.minimum(updates_mask, 1)
    updates_mask_3d = tf.minimum(updates_mask_3d, 1)

    updated_sequence = (((1 - updates_mask_3d) * sequence) +
                        (updates_mask_3d * updates))
    if not depth_dimension:
        updated_sequence = tf.squeeze(updated_sequence, -1)

    return updated_sequence, updates_mask


def get_bert_config(model_size, vocab_size):
    if model_size == 'large':
        args = {'hidden_size': 1024, 'num_hidden_layers': 24}
    elif model_size == 'base':
        args = {'hidden_size': 768, 'num_hidden_layers': 12}
    elif model_size == 'small':
        args = {'hidden_size': 256, 'num_hidden_layers': 12}
    else:
        raise ValueError(
            'Unknown `model_size` %s. Pick one from '
            '(`small`, `base`, `large`).' % model_size)
    args['vocab_size'] = vocab_size
    args['num_attention_heads'] = max(1, args['hidden_size'] // 64)
    args['intermediate_size'] = 4 * args['hidden_size']
    return BERTConfig.from_dict(args)


def get_generator_config(config: PretrainingConfig,
                         bert_config: BERTConfig):
    '''Get model config for the generator network.'''
    gen_config = copy.deepcopy(bert_config)
    gen_config.hidden_size = int(round(
        bert_config.hidden_size * config.generator_hidden_size))
    gen_config.num_hidden_layers = int(round(
        bert_config.num_hidden_layers * config.generator_layers))
    gen_config.intermediate_size = 4 * gen_config.hidden_size
    gen_config.num_attention_heads = max(1, gen_config.hidden_size // 64)
    return gen_config
