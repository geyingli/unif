""" Convolutional neural network on texture analysis. """

from ...third import tf
from .._base_._base_ import BaseEncoder
from .. import util


class TextCNNEncoder(BaseEncoder):
    def __init__(self,
                 vocab_size,
                 filter_sizes,
                 num_channels,
                 is_training,
                 input_ids,
                 scope="text_cnn",
                 embedding_size=256,
                 dropout_prob=0.1,
                 trainable=True,
                 **kwargs):

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        if isinstance(filter_sizes, str):
            filter_sizes = filter_sizes.split(",")
        assert isinstance(filter_sizes, list), (
            "`filter_sizes` should be a list of integers or a string "
            "seperated with commas.")

        with tf.variable_scope(scope):
            with tf.variable_scope("embeddings"):

                embedding_table = kwargs.get("tilda_embeddings")
                if embedding_table is None:
                    embedding_table = tf.get_variable(
                        name="word_embeddings",
                        shape=[vocab_size, embedding_size],
                        initializer=util.create_initializer(0.02),
                        dtype=tf.float32,
                        trainable=trainable)

                flat_input_ids = tf.reshape(input_ids, [-1])
                output = tf.gather(
                    embedding_table, flat_input_ids, name="embedding_look_up")
                output = tf.reshape(
                    output, [batch_size, max_seq_length, embedding_size])

                output_expanded = tf.expand_dims(output, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv_%s" % filter_size):

                    # Convolution Layer
                    W = tf.get_variable(
                        name="W",
                        shape=[int(filter_size), embedding_size, 1, num_channels],
                        initializer=tf.truncated_normal_initializer(0.1),
                        dtype=tf.float32,
                        trainable=trainable)
                    b = tf.get_variable(
                        name="b",
                        shape=[num_channels],
                        initializer=tf.constant_initializer(0.1),
                        dtype=tf.float32,
                        trainable=trainable)
                    conv = tf.nn.conv2d(
                        output_expanded, W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, max_seq_length - int(filter_size) + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool")
                    pooled_outputs.append(pooled)

            num_channels_total = num_channels * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [batch_size, num_channels_total])

            with tf.name_scope("dropout"):
                self.pooled_output = util.dropout(h_pool_flat, dropout_prob)

    def get_pooled_output(self):
        """ Returns a tensor with shape [batch_size, hidden_size]. """
        return self.pooled_output


def get_decay_power():
    decay_power = {
        "/embeddings": 2,
        "/conv_": 1,
        "cls/": 0,
    }
    return decay_power
