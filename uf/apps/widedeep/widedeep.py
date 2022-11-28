""" Wide & Deep structure. """

import math

from ...third import tf
from .._base_._base_ import BaseDecoder
from .. import util


class WideDeepClsDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 wide_ids,
                 wide_weights,
                 wide_length,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 scope="cls/seq_relationship",
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("return_hidden"):
            self.tensors["hidden"] = input_tensor

        hidden_size = input_tensor.shape.as_list()[-1]
        feature_size = wide_ids.shape.as_list()[-1]
        with tf.variable_scope("wide"):
            feature_embeddings = tf.get_variable(
                name="feature_embeddings",
                shape=[feature_size + 1, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            wide_output = tf.gather(feature_embeddings, wide_ids)               # [B, N, H]
            wide_output *= tf.expand_dims(wide_weights, 1)

        with tf.variable_scope("wide_and_deep"):
            deep_output = tf.expand_dims(input_tensor, -1)                      # [B, H, 1]
            attention_scores = tf.matmul(wide_output, deep_output)              # [B, N, 1]
            attention_scores = tf.transpose(attention_scores, [0, 2, 1])        # [B, 1, N]
            attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(hidden_size))
            feature_mask = tf.cast(tf.sequence_mask(wide_length, feature_size), tf.float32)    # [B, N]
            feature_mask = tf.expand_dims(feature_mask, 1)                      # [B, 1, N]
            attention_scores += (1.0 - feature_mask) * -10000.0
            attention_matrix = tf.nn.softmax(attention_scores, axis=-1)
            attention_output = tf.matmul(attention_matrix, wide_output)         # [B, 1, H]
            attention_output = attention_output[:, 0, :]                        # [B, H]
            # attention_output = util.dropout(
            #     attention_output, hidden_dropout_prob)
            input_tensor = util.layer_norm(attention_output + input_tensor, trainable=trainable)

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

            output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            self.tensors["preds"] = tf.argmax(logits, axis=-1)
            self.tensors["probs"] = tf.nn.softmax(logits, axis=-1, name="probs")

            per_example_loss = util.cross_entropy(logits, label_ids, label_size, **kwargs)
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.tensors["losses"] = per_example_loss
            self.train_loss = tf.reduce_mean(per_example_loss)


class WideDeepRegDecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 wide_ids,
                 wide_weights,
                 wide_length,
                 label_floats,
                 label_size=1,
                 sample_weight=None,
                 scope="reg",
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        if kwargs.get("return_hidden"):
            self.tensors["hidden"] = input_tensor

        hidden_size = input_tensor.shape.as_list()[-1]
        feature_size = wide_ids.shape.as_list()[-1]
        with tf.variable_scope("wide"):
            feature_embeddings = tf.get_variable(
                name="feature_embeddings",
                shape=[feature_size + 1, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            wide_output = tf.gather(feature_embeddings, wide_ids)  # [B, N, H]

        with tf.variable_scope("wide_and_deep"):
            deep_output = tf.expand_dims(input_tensor, -1)  # [B, H, 1]
            attention_scores = tf.matmul(wide_output, deep_output)  # [B, N, 1]
            attention_scores = tf.transpose(attention_scores, [0, 2, 1])  # [B, 1, N]
            attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(hidden_size))
            feature_mask = tf.cast(tf.sequence_mask(wide_length, feature_size), tf.float32)    # [B, N]
            feature_mask = tf.expand_dims(feature_mask, 1)  # [B, 1, N]
            attention_scores += (1.0 - feature_mask) * -10000.0
            attention_matrix = tf.nn.softmax(attention_scores, axis=-1)
            attention_output = tf.matmul(attention_matrix, wide_output)  # [B, 1, H]
            attention_output = attention_output[:, 0, :]  # [B, H]
            # attention_output = util.dropout(
            #     attention_output, hidden_dropout_prob)
            input_tensor = util.layer_norm(attention_output + input_tensor, trainable=trainable)

        with tf.variable_scope(scope):
            intermediate_output = tf.layers.dense(
                input_tensor,
                label_size * 4,
                use_bias=False,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable,
            )
            probs = tf.layers.dense(
                intermediate_output,
                label_size,
                use_bias=False,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable,
                name="probs",
            )

            self.tensors["probs"] = probs

            per_example_loss = util.mean_squared_error(probs, label_floats, **kwargs)
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.tensors["losses"] = per_example_loss
            self.train_loss = tf.reduce_mean(per_example_loss)


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": num_hidden_layers + 2,
        "wide/": 2,
        "wide_and_deep/": 1,
        "/pooler/": 1,
        "cls/": 0,
        "reg": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/layer_%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power

