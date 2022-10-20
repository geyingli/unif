from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

from .._base_._base_ import BaseEncoder
from .. import util
from ...third import tf


class BiRNNEncoder(BaseEncoder):
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
        half_hidden_size = hidden_size // 2
        self.rnn_core = rnn_core

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]


        with tf.variable_scope(scope):

            # embedding
            embedding_output, _ = util.embedding_lookup(
                input_ids=input_ids,
                vocab_size=vocab_size,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                embedding_size=hidden_size,
                word_embedding_name="word_embeddings",
                tilda_embeddings=kwargs.get("tilda_embeddings"),
                trainable=trainable,
            )

            # rnn core
            if rnn_core == "rnn":
                cell_fw = rnn_cell.BasicRNNCell(num_units=half_hidden_size, trainable=trainable)
                cell_bw = rnn_cell.BasicRNNCell(num_units=half_hidden_size, trainable=trainable)
            elif rnn_core == "lstm":
                cell_fw = rnn_cell.LSTMCell(num_units=half_hidden_size, trainable=trainable)
                cell_bw = rnn_cell.LSTMCell(num_units=half_hidden_size, trainable=trainable)
            elif rnn_core == "gru":
                cell_fw = rnn_cell.GRUCell(num_units=half_hidden_size, trainable=trainable)
                cell_bw = rnn_cell.GRUCell(num_units=half_hidden_size, trainable=trainable)
            dropout_cell_fw = rnn_cell.DropoutWrapper(cell_fw, state_keep_prob=1 - dropout_rate)
            dropout_cell_bw = rnn_cell.DropoutWrapper(cell_bw, state_keep_prob=1 - dropout_rate)

            # inputs: [batch_size, max_seq_length, hidden_size]
            # outputs: ([batch_size, max_seq_length, half_hidden_size], [batch_size, max_seq_length, half_hidden_size])
            outputs, self.last_states = rnn.bidirectional_dynamic_rnn(
                cell_fw=dropout_cell_fw,
                cell_bw=dropout_cell_bw,
                inputs=embedding_output,
                sequence_length=seq_length,
                dtype=tf.float32,
            )
            self.outputs = tf.concat(outputs, axis=2)

    def get_pooled_output(self):
        return self.outputs[:, 0, :]

    def get_sequence_output(self):
        return self.outputs


def get_decay_power():
    return {
        "word_embeddings": 2,
        "/bidirectional_rnn/": 1,
        "cls/": 0,
    }