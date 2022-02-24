""" Conditional Random Field (CRF). """

import numpy as np

from ..thirdparty import tf
from tensorflow.python.framework import smart_cond as smart
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from .base import BaseDecoder
from . import util


class CRFDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 input_mask,
                 label_ids,
                 label_size=5,
                 sample_weight=None,
                 scope="cls/sequence",
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        seq_length = input_tensor.shape.as_list()[-2]
        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                "output_bias",
                shape=[label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)

            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, seq_length, label_size])

            with tf.variable_scope("crf"):
                input_length = tf.reduce_sum(input_mask, axis=-1)
                per_example_loss, transition_matrix = crf_log_likelihood(
                    input_tensor=logits,
                    tag_indices=label_ids,
                    sequence_lengths=input_length)
                per_example_loss = - per_example_loss
                if sample_weight is not None:
                    per_example_loss *= tf.cast(
                        sample_weight, dtype=tf.float32)
                self.total_loss = tf.reduce_mean(per_example_loss)
                self._tensors["losses"] = per_example_loss
                self._tensors["preds"] = tf.argmax(logits, axis=-1)
                self._tensors["logits"] = logits
                self._tensors["transition_matrix"] = transition_matrix


def crf_log_likelihood(input_tensor,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
    """ Computes the log-likelihood of tag sequences in a CRF.

    Args:
        input_tensor: A [batch_size, max_seq_len, num_tags] tensor of unary
          potentials to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
          compute the log-likelihood.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix, if available.

    Returns:
        log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
          each example, given the sequence of tag indices.
        transition_params: A [num_tags, num_tags] transition matrix. This is either
          provided by the caller or created in this function.
    """
    # Get shape information.
    num_tags = util.get_shape_list(input_tensor)[2]

    # Get the transition matrix if not provided.
    if transition_params is None:
        transition_params = tf.get_variable(
            "transitions", [num_tags, num_tags])

    sequence_scores = crf_sequence_score(
        input_tensor, tag_indices, sequence_lengths,
        transition_params)
    log_norm = crf_log_norm(input_tensor, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """ Computes the unnormalized score for a tag sequence.

    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary
          potentials to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for
          which we compute the unnormalized score.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.

    Returns:
        sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of the single tag.
    def _single_seq_fn():
      batch_size = tf.shape(inputs, out_type=tag_indices.dtype)[0]
      example_inds = tf.reshape(
          tf.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
      sequence_scores = tf.gather_nd(
          tf.squeeze(inputs, [1]),
          tf.concat([example_inds, tag_indices], axis=1))
      sequence_scores = tf.where(tf.less_equal(sequence_lengths, 0),
                                 tf.zeros_like(sequence_scores),
                                 sequence_scores)
      return sequence_scores

    def _multi_seq_fn():
      # Compute the scores of the given tag sequence.
      unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
      binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                       transition_params)
      sequence_scores = unary_scores + binary_scores
      return sequence_scores

    return smart.smart_cond(
        pred=tf.equal(
            util.get_shape_list(inputs)[1] or tf.shape(inputs)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """ Computes the unary scores of tag sequences.

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(
        tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    # Use int32 or int64 based on tag_indices" dtype.
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = tf.sequence_mask(sequence_lengths,
                             maxlen=tf.shape(tag_indices)[1],
                             dtype=tf.float32)

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """ Computes the binary scores of tag sequences.

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    # Get shape information.
    num_tags = transition_params.get_shape()[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = tf.slice(tag_indices, [0, 0],
                                        [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = \
        start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = tf.gather(flattened_transition_params,
                                     flattened_transition_indices)

    masks = tf.sequence_mask(sequence_lengths,
                                    maxlen=tf.shape(tag_indices)[1],
                                    dtype=tf.float32)
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """ Computes the normalization for a CRF.

    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.

    Returns:
        log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp
    # over the "initial state" (the unary potentials).
    def _single_seq_fn():
      log_norm = tf.reduce_logsumexp(first_input, [1])
      # Mask `log_norm` of the sequences with length <= zero.
      log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                          tf.zeros_like(log_norm),
                          log_norm)
      return log_norm

    def _multi_seq_fn():
      """Forward computation of alpha values."""
      rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])

      # Compute the alpha values in the forward algorithm in order to get the
      # partition function.
      forward_cell = CrfForwardRnnCell(transition_params)
      # Sequence length is not allowed to be less than zero.
      sequence_lengths_less_one = tf.maximum(
          tf.constant(0, dtype=sequence_lengths.dtype),
          sequence_lengths - 1)
      _, alphas = rnn.dynamic_rnn(
          cell=forward_cell,
          inputs=rest_of_input,
          sequence_length=sequence_lengths_less_one,
          initial_state=first_input,
          dtype=tf.float32)
      log_norm = tf.reduce_logsumexp(alphas, [1])
      # Mask `log_norm` of the sequences with length <= zero.
      log_norm = tf.where(tf.less_equal(sequence_lengths, 0),
                          tf.zeros_like(log_norm),
                          log_norm)
      return log_norm

    return smart.smart_cond(
        pred=tf.equal(
            util.get_shape_list(inputs)[1] or tf.shape(inputs)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


class CrfForwardRnnCell(rnn_cell.RNNCell):
  """ Computes the alpha values in a linear-chain CRF.

  See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
  """

  def __init__(self, transition_params):
    """Initialize the CrfForwardRnnCell.

    Args:
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
    """
    self._transition_params = tf.expand_dims(transition_params, 0)
    self._num_tags = util.get_shape_list(transition_params)[0]

  @property
  def state_size(self):
      return self._num_tags

  @property
  def output_size(self):
      return self._num_tags

  def __call__(self, inputs, state, scope=None):
      """ Build the CrfForwardRnnCell.

      Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
            values.
          scope: Unused variable scope of this cell.

      Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
            values containing the new alpha values.
      """
      state = tf.expand_dims(state, 2)

      # This addition op broadcasts self._transitions_params along the zeroth
      # dimension and state along the second dimension. This performs the
      # multiplication of previous alpha values and the current binary
      # potentials in log space.
      transition_scores = state + self._transition_params
      new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])

      # Both the state and the output of this RNN cell contain the alphas
      # values. The output value is currently unused and simply satisfies the
      # RNN API. This could be useful in the future if we need to compute
      # marginal probabilities, which would require the accumulated alpha
      # values at every time step.
      return new_alphas, new_alphas


def viterbi_decode(score, transition_params):
    """ Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score
