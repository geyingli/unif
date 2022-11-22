""" Commonly-used modeling methods. """

import math
import numpy as np

from ..third import tf


class HParams:
    """ Hyparameters. """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
    def set_hparam(self, k, v):
        self.__dict__[k] = v


def gelu(num):
    """ Gaussian Error Linear Unit, a smoother version of the RELU.
    paper: https://arxiv.org/abs/1606.08415 """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
                                (num + 0.044715 * tf.pow(num, 3)))))
    return num * cdf


def get_activation(activation_string):
    """ Returns activation function given string. """
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    if act == "relu":
        return tf.nn.relu
    if act == "gelu":
        return gelu
    if act == "tanh":
        return tf.tanh
    raise ValueError("Unsupported activation: %s" % act)


def xavier_initializer(uniform=True,
                       factor=1.0,
                       mode="FAN_AVG",
                       seed=None,
                       dtype=tf.float32):
    """ Returns an initializer performing "Xavier" initialization for weights.

    This function implements the weight initialization from:
    Xavier Glorot and Yoshua Bengio (2010):
             [Understanding the difficulty of training deep feedforward neural
             networks. International conference on artificial intelligence and
             statistics.](
             http://www.jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

    This initializer is designed to keep the scale of the gradients roughly
    the     same in all layers. In uniform distribution this ends up being
    the range: `x = sqrt(6. / (in + out)); [-x, x]` and for normal
    distribution a standard deviation of `sqrt(2. / (in + out))` is used.

    Args:
        uniform: Whether to use uniform or normal distributed random
          initialization.
        seed: A Python integer. Used to create random seeds. See
          `tf.compat.v1.set_random_seed` for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer for a weight matrix.
    """

    if not dtype.is_floating:
        raise TypeError(
            "Cannot create initializer for non-floating point type.")
    if mode not in ["FAN_IN", "FAN_OUT", "FAN_AVG"]:
        raise TypeError("Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]", mode)

    def _initializer(shape, dtype=dtype, partition_info=None):
      """Initializer function."""
      if not dtype.is_floating:
          raise TypeError(
              "Cannot create initializer for non-floating point type.")
      # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
      # This is the right thing for matrix multiply and convolutions.
      if shape:
          fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
          fan_out = float(shape[-1])
      else:
          fan_in = 1.0
          fan_out = 1.0
      for dim in shape[:-2]:
          fan_in *= float(dim)
          fan_out *= float(dim)
      if mode == "FAN_IN":
          # Count only number of input connections.
          n = fan_in
      elif mode == "FAN_OUT":
          # Count only number of output connections.
          n = fan_out
      elif mode == "FAN_AVG":
          # Average number of inputs and output connections.
          n = (fan_in + fan_out) / 2.0
      if uniform:
          # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
          limit = math.sqrt(3.0 * factor / n)
          return tf.random_uniform(
              shape, -limit, limit, dtype, seed=seed)
      else:
          # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
          trunc_stddev = math.sqrt(1.3 * factor / n)
          return tf.truncated_normal(
              shape, 0.0, trunc_stddev, dtype, seed=seed)

    return _initializer


def embedding_lookup(
    input_ids,
    vocab_size,
    batch_size,
    max_seq_length,
    embeddings=None,
    embedding_size=128,
    initializer_range=0.02,
    word_embedding_name="word_embeddings",
    dtype=tf.float32,
    trainable=True,
):
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    if embeddings is not None:
        embedding_table = embeddings
    else:
        embedding_table = tf.get_variable(
            name=word_embedding_name,
            shape=[vocab_size, embedding_size],
            initializer=create_initializer(initializer_range),
            dtype=dtype,
            trainable=trainable,
        )

    flat_input_ids = tf.reshape(input_ids, [-1])
    output = tf.gather(embedding_table, flat_input_ids, name="embedding_look_up")
    output = tf.reshape(output, [batch_size, max_seq_length, embedding_size])
    return output, embedding_table


def layer_norm(input_tensor,
               center=True,
               scale=True,
               activation_fn=None,
               variables_collections=None,
               outputs_collections=None,
               begin_norm_axis=-1,
               begin_params_axis=-1,
               trainable=True,
               scope="LayerNorm",
               reuse=None):
    """ Runs layer normalization on the last dimension of the tensor.

    Args:
      input_tensor: A tensor having rank `R`. The normalization is performed
        over axes `begin_norm_axis ... R - 1` and centering and scaling
        parameters are calculated over `begin_params_axis ... R - 1`.
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      activation_fn: Activation function, default set to None to skip it and
        maintain a linear activation.
      variables_collections: Optional collections for the variables.
      outputs_collections: Collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      begin_norm_axis: The first normalization dimension: normalization will
        be performed along dimensions `begin_norm_axis : rank(input_tensor)`
      begin_params_axis: The first parameter (beta, gamma) dimension: scale
        and centering parameters will have dimensions
        `begin_params_axis : rank(input_tensor)` and will be broadcast with
        the normalized inputs accordingly.
      scope: Optional scope for `variable_scope`.

    Returns:
      A `Tensor` representing the output of the operation, having the same
      shape and dtype as `input_tensor`.

    Raises:
      ValueError: If the rank of `input_tensor` is not known at graph build
        time, or if `input_tensor.shape[begin_params_axis:]` is not fully
        defined at graph build time.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = input_tensor.shape
        inputs_rank = inputs_shape.ndims
        if inputs_rank is None:
            raise ValueError("Inputs %s has undefined rank."
                             % input_tensor.name)
        dtype = input_tensor.dtype.base_dtype
        if begin_norm_axis < 0:
            begin_norm_axis = inputs_rank + begin_norm_axis
        if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
            raise ValueError(
                "begin_params_axis (%d) and begin_norm_axis (%d) "
                "must be < rank(inputs) (%d)"
                % (begin_params_axis, begin_norm_axis, inputs_rank))
        params_shape = inputs_shape[begin_params_axis:]
        if not params_shape.is_fully_defined():
            raise ValueError(
                "Inputs %s: shape(inputs)[%s:] is not fully defined: %s" %
                (input_tensor.name, begin_params_axis, inputs_shape))

        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable(
                "beta", shape=params_shape,
                dtype=dtype,
                initializer=tf.zeros_initializer(),
                trainable=trainable)
        if scale:
            gamma = tf.get_variable(
                "gamma",
                shape=params_shape,
                dtype=dtype,
                initializer=tf.ones_initializer(),
                trainable=trainable)
        # By default, compute the moments across all the dimensions except the
        # one with index 0.
        norm_axes = list(range(begin_norm_axis, inputs_rank))
        mean, variance = tf.nn.moments(input_tensor, norm_axes, keep_dims=True)

        # Compute layer normalization using the batch_normalization function.
        # Note that epsilon must be increased for float16 due to the limited
        # representable range.
        variance_epsilon = 1e-12 if dtype != tf.float16 else 1e-3
        outputs = tf.nn.batch_normalization(
            input_tensor, mean, variance,
            offset=beta, scale=gamma,
            variance_epsilon=variance_epsilon)

        outputs.set_shape(inputs_shape)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def dropout(input_tensor, dropout_prob):
    """ A more intuitive dropout function. """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    try:
        output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    except:
        output = tf.nn.dropout(input_tensor, keep_prob=1.0 - dropout_prob)
    return output


def layer_norm_and_dropout(input_tensor, dropout_prob, trainable=True):
    """ Runs layer normalization followed by dropout. """
    output_tensor = layer_norm(input_tensor, trainable=trainable)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def cross_entropy(logits, label_ids, label_size, **kwargs):
    """ Cross Entropy Loss for single-label classification. """

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    one_hot_labels = tf.one_hot(label_ids, depth=label_size, dtype=tf.float32)

    # focal loss
    if kwargs.get("focal_loss"):
        gamma = kwargs.get("gamma", 1.0)
        log_probs *= tf.pow(1 - tf.nn.softmax(logits, axis=-1), gamma)

    # label smoothing
    if kwargs.get("label_smoothing"):
        smoothing_rate = kwargs.get("smoothing_rate", 0.1)
        one_hot_labels = ((1 - smoothing_rate) * one_hot_labels) + (smoothing_rate / label_size)

    per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

    # training signal annealing
    thresh = kwargs.get("tsa_thresh")
    if thresh is not None:
        assert isinstance(thresh, float), "`tsa_thresh` must be a float between 0 and 1."
        probs = tf.nn.softmax(logits, axis=-1)
        uncertainty = tf.reduce_sum(probs * tf.log(probs), axis=-1)
        uncertainty /= tf.log(1 / label_size)
        per_example_loss *= tf.cast(tf.greater(uncertainty, thresh), dtype=tf.float32)

    # cut extreme loss
    thresh = kwargs.get("conf_thresh")
    if thresh is not None:
        assert isinstance(thresh, float), "`conf_thresh` must be a float between 0 and 1."
        largest_prob = tf.reduce_max(tf.exp(log_probs), axis=-1)
        per_example_loss *= tf.cast(tf.less(largest_prob, thresh), dtype=tf.float32)

    return per_example_loss


def sigmoid_cross_entropy(logits, label_ids, label_size=None, label_weight=None, keep_dims=False, **kwargs):
    """ Cross Entropy Loss for multi-label classification. """

    per_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(label_ids, dtype=tf.float32))

    # label_weight
    if label_weight is not None:
        label_weight = tf.constant(label_weight, dtype=tf.float32)
        label_weight = tf.reshape(label_weight, [1, label_size])
        per_label_loss *= label_weight

    if keep_dims:
        return per_label_loss
    per_example_loss = tf.reduce_sum(per_label_loss, axis=-1)
    return per_example_loss


def bidirectional_kl_divergence(p, q):
    """ Bidirectional Kullback-Leibler divergence. """
    return kl_divergence(p, q) + kl_divergence(q, p)


def kl_divergence(p, q):
    """ Kullback-Leibler divergence. """
    per_example_loss = tf.reduce_sum(p * (tf.log(p) - tf.log(q)), axis=-1)
    return per_example_loss


def mean_squared_error(logits, label_floats, **kwargs):
    """ MSE loss for regression. """
    per_float_loss = tf.square(logits - label_floats)
    per_example_loss = tf.reduce_sum(per_float_loss, axis=-1)
    return per_example_loss


def info_nce(p, q, tau=1.0):
    """ InfoNCE loss. """
    batch_size = get_shape_list(p)[0]
    sim_matrix = cosine_similarity(p, q) / tau
    nom = tf.reduce_sum(tf.eye(batch_size) * sim_matrix, axis=-1)
    denom = tf.reduce_sum(sim_matrix, axis=-1)
    per_example_loss = - tf.log(nom / denom)
    return per_example_loss


def cosine_similarity(p, q):
    """ Cosine similarity. """
    dot_product = tf.matmul(p, q, transpose_b=True)
    l2_norm_p = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(p), axis=-1)), [-1, 1])
    l2_norm_q = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(q), axis=-1)), [1, -1])
    sim_matrix = dot_product / (l2_norm_p * l2_norm_q)
    return sim_matrix


def create_initializer(initializer_range=0.02):
    """ Creates a truncated_normal_initializer with the given range. """
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor, expected_rank=None, name=None):
    """ Returns a list of the shape of tensor, preferring static dimensions. """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """ Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix). """
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError(
            "Input tensor must have at least rank 2. Shape = %s"
            % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list,
                        original_shape=None):
    """ Reshapes a rank 2 tensor back to its original rank >= 2 tensor. """
    if len(orig_shape_list) == 2:
        return output_tensor

    if not original_shape:
        original_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = original_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """ Raises an exception if the tensor rank is not of the expected rank. """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, int):
        expected_rank_dict[expected_rank] = True
    else:
        for rank in expected_rank:
            expected_rank_dict[rank] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor %s in scope %s, the actual rank "
            "%d (shape = %s) is not equal to the expected rank %s"
            % (name,
               scope_name,
               actual_rank,
               str(tensor.shape),
               str(expected_rank)))
