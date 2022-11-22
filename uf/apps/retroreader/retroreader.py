""" Retrospective Reader (Retro-Reader). """

import math
import numpy as np

from ...third import tf
from .._base_._base_ import BaseDecoder
from .. import util


class RetroReaderDecoder(BaseDecoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 sketchy_encoder,
                 intensive_encoder,
                 query_mask,
                 label_ids,
                 has_answer,
                 sample_weight=None,
                 scope="retro_reader",
                 matching_mechanism="cross-attention",
                 beta_1=0.5,
                 beta_2=0.5,
                 threshold=1.0,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("return_hidden"):
            self.tensors["hidden"] = intensive_encoder.get_sequence_output()[:, 0, :]

        # verifier
        with tf.variable_scope(scope):

            # sketchy reading module
            with tf.variable_scope("sketchy/prediction"):
                sketchy_output = sketchy_encoder.get_pooled_output()
                hidden_size = sketchy_output.shape.as_list()[-1]

                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[2, hidden_size],
                    initializer=util.create_initializer(
                        bert_config.initializer_range),
                    trainable=trainable)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[2],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable)

                output_layer = util.dropout(
                    sketchy_output, bert_config.hidden_dropout_prob \
                        if is_training else 0.0)
                logits = tf.matmul(
                    output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

                per_example_loss = util.cross_entropy(logits, has_answer, 2, **kwargs)
                if sample_weight is not None:
                    per_example_loss *= sample_weight
                self.tensors["sketchy_losses"] = per_example_loss
                sketchy_loss = tf.reduce_mean(per_example_loss)

                score_ext = logits[:, 1] - logits[:, 0]

            # intensive reading module
            with tf.variable_scope("intensive"):
                H = intensive_encoder.get_sequence_output()
                H_Q = H * tf.cast(
                    tf.expand_dims(query_mask, axis=-1), tf.float32)
                (batch_size, max_seq_length, hidden_size) = \
                    util.get_shape_list(H)

                # cross-attention
                if matching_mechanism == "cross-attention":
                    with tf.variable_scope("cross_attention"):
                        attention_mask = self.create_attention_mask_from_input_mask(
                            query_mask, batch_size, max_seq_length)
                        H_prime, _ = self.attention_layer(
                            from_tensor=H,
                            to_tensor=H_Q,
                            attention_mask=attention_mask,
                            num_attention_heads=bert_config.num_attention_heads,
                            size_per_head=hidden_size // bert_config.num_attention_heads,
                            attention_probs_dropout_prob=bert_config.hidden_dropout_prob,
                            initializer_range=bert_config.initializer_range,
                            do_return_2d_tensor=False,
                            batch_size=batch_size,
                            from_max_seq_length=max_seq_length,
                            to_max_seq_length=max_seq_length,
                            trainable=trainable)

                # matching-attention
                elif matching_mechanism == "matching-attention":
                    with tf.variable_scope("matching_attention"):
                        output_weights = tf.get_variable(
                            "output_weights",
                            shape=[hidden_size, hidden_size],
                            initializer=util.create_initializer(
                                bert_config.initializer_range),
                            trainable=trainable)
                        output_bias = tf.get_variable(
                            "output_bias",
                            shape=[hidden_size],
                            initializer=tf.zeros_initializer(),
                            trainable=trainable)
                        trans = tf.matmul(
                            H_Q, tf.tile(
                                tf.expand_dims(output_weights, axis=0),
                                [batch_size, 1, 1]),
                            transpose_b=True)
                        trans = tf.nn.bias_add(trans, output_bias)
                        M = tf.nn.softmax(
                            tf.matmul(H, trans, transpose_b=True), axis=-1)
                        H_prime = tf.matmul(M, H_Q)

                with tf.variable_scope("prediction"):
                    output_weights = tf.get_variable(
                        "output_weights",
                        shape=[2, hidden_size],
                        initializer=util.create_initializer(bert_config.initializer_range),
                        trainable=trainable)
                    output_bias = tf.get_variable(
                        "output_bias",
                        shape=[2],
                        initializer=tf.zeros_initializer(),
                        trainable=trainable)

                    output_layer = util.dropout(H_prime, bert_config.hidden_dropout_prob if is_training else 0.0)
                    output_layer = tf.reshape(output_layer, [batch_size * max_seq_length, hidden_size])
                    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                    logits = tf.nn.bias_add(logits, output_bias)
                    logits = tf.reshape(logits, [batch_size, max_seq_length, 2])
                    logits = tf.transpose(logits, [0, 2, 1])
                    probs = tf.nn.softmax(logits, axis=-1, name="probs")

                    self.tensors["mrc_probs"] = probs
                    self.tensors["mrc_preds"] = tf.argmax(logits, axis=-1)

                    per_example_loss = (
                        0.5 * util.cross_entropy(logits[:, 0, :], label_ids[:, 0], max_seq_length, **kwargs) +
                        0.5 * util.cross_entropy(logits[:, 1, :], label_ids[:, 1], max_seq_length, **kwargs)
                    )
                    if sample_weight is not None:
                        per_example_loss *= sample_weight
                    intensive_loss = tf.reduce_mean(per_example_loss)
                    self.tensors["intensive_losses"] = per_example_loss

                    score_has = tf.norm(probs[:, 0, 1:] + probs[:, 1, 1:], np.inf, axis=-1)
                    score_null = probs[:, 0, 0] + probs[:, 1, 0]
                    score_diff = score_has - score_null

            # rear verification
            v = beta_1 * score_diff + beta_2 * score_ext
            self.tensors["verifier_preds"] = tf.cast(tf.greater(v, threshold), tf.int32)
            self.tensors["verifier_probs"] = v

            self.train_loss = sketchy_loss + intensive_loss

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


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": num_hidden_layers + 2,
        "/pooler/": 1,
        "retro_reader/": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/layer_%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power
