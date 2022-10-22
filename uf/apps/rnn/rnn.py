from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

from .._base_._base_ import BaseEncoder
from .. import util
from ...third import tf


class RNNEncoder(BaseEncoder):
    def __init__(
        self,
        is_training,
        input_ids,
        seq_length,
        vocab_size,
        rnn_core="lstm",
        hidden_size=128,
        scope="rnn",
        trainable=True,
        **kwargs,
    ):
        dropout_rate = 0.0
        if is_training:
            dropout_rate = 0.1
        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        self.rnn_core = rnn_core

        with tf.variable_scope(scope):

            # embedding
            embedding_output, _ = util.embedding_lookup(
                input_ids=input_ids,
                vocab_size=vocab_size,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                embeddings=kwargs.get("tilda_embeddings"),
                embedding_size=hidden_size,
                word_embedding_name="word_embeddings",
                trainable=trainable,
            )

            # rnn core
            if rnn_core == "rnn":
                cell = rnn_cell.BasicRNNCell(num_units=hidden_size, trainable=trainable)
            elif rnn_core == "lstm":
                cell = rnn_cell.LSTMCell(num_units=hidden_size, trainable=trainable)
            elif rnn_core == "gru":
                cell = rnn_cell.GRUCell(num_units=hidden_size, trainable=trainable)
            dropout_cell = rnn_cell.DropoutWrapper(cell, state_keep_prob=1 - dropout_rate)

            # inputs: [batch_size, max_seq_length, hidden_size]
            # outputs: [batch_size, max_seq_length, hidden_size]
            self.outputs, self.last_states = rnn.dynamic_rnn(
                cell=dropout_cell,
                inputs=embedding_output,
                sequence_length=seq_length,
                dtype=tf.float32,
            )

    def get_pooled_output(self):
        if self.rnn_core == "lstm":
            return self.last_states[-1]     # ([batch_size, hidden_size], [batch_size, hidden_size])
        return self.last_states             # [batch_size, hidden_size]

    def get_sequence_output(self):
        return self.outputs


def get_decay_power():
    return {
        "word_embeddings": 2,
        "/rnn/": 1,
        "cls/": 0,
    }