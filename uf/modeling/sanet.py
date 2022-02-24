""" Sentence attention decoder. """

import math

from ..thirdparty import tf
from .base import BaseDecoder
from . import util


class SANetDecoder(BaseDecoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_tensor,
                 sa_mask,
                 label_ids,
                 sample_weight=None,
                 scope="sanet",
                 alpha=0.5,
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        shape = util.get_shape_list(input_tensor)
        batch_size = shape[0]
        seq_length = shape[1]
        hidden_size = shape[2]
        sa_mask = tf.reshape(sa_mask, [batch_size, seq_length, seq_length])
        with tf.variable_scope(scope):
            with tf.variable_scope("sentence_attention"):
                (sa_output, _) = self.attention_layer(
                    from_tensor=input_tensor,
                    to_tensor=input_tensor,
                    attention_mask=sa_mask,
                    num_attention_heads=bert_config.num_attention_heads,
                    size_per_head=\
                        hidden_size // bert_config.num_attention_heads,
                    attention_probs_dropout_prob=\
                        bert_config.hidden_dropout_prob,
                    initializer_range=bert_config.initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_max_seq_length=seq_length,
                    to_max_seq_length=seq_length,
                    trainable=trainable)

            with tf.variable_scope("cls/mrc"):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[2, hidden_size],
                    initializer=util.create_initializer(initializer_range),
                    trainable=trainable)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[2],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable)

            output_layer = alpha * sa_output + (1 - alpha) * input_tensor
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, seq_length, 2])
            logits = tf.transpose(logits, [0, 2, 1])
            probs = tf.nn.softmax(logits, axis=-1, name="probs")
            self._tensors["probs"] = probs
            self._tensors["preds"] = tf.argmax(logits, axis=-1)

            start_one_hot_labels = tf.one_hot(
                label_ids[:, 0], depth=seq_length, dtype=tf.float32)
            end_one_hot_labels = tf.one_hot(
                label_ids[:, 1], depth=seq_length, dtype=tf.float32)
            start_log_probs = tf.nn.log_softmax(logits[:, 0, :], axis=-1)
            end_log_probs = tf.nn.log_softmax(logits[:, 1, :], axis=-1)
            per_example_loss = (
                - 0.5 * tf.reduce_sum(
                    start_one_hot_labels * start_log_probs, axis=-1)
                - 0.5 * tf.reduce_sum(
                    end_one_hot_labels * end_log_probs, axis=-1))
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.total_loss = tf.reduce_mean(per_example_loss)
            self._tensors["losses"] = per_example_loss

    def attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        num_attention_heads=12,
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
            name="query",
            kernel_initializer=util.create_initializer(initializer_range),
            trainable=trainable)

        # key_layer = [B*T, N*H]
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=util.create_initializer(initializer_range),
            trainable=trainable)

        # value_layer = [B*T, N*H]
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
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

        # Take the dot product between "query" and "key" to get the raw
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
