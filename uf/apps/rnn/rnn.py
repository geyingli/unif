from .._base_._base_ import BaseEncoder
from .. import util
from ...third import tf
from tf.contrib.rnn import RNNCell, LSTMCell, GRUCell, DropoutWrapper, MultiRNNCell


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
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        with tf.variable_scope(scope):

            # embedding
            embedding_output, _ = util.embedding_lookup(
                input_ids=input_ids,
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_length=max_seq_length,
                embedding_size=hidden_size,
                initializer_range=initializer_range,
                word_embedding_name="word_embeddings",
                tilda_embeddings=kwargs.get("tilda_embeddings"),
                trainable=trainable,
            )

            # backbone
            if rnn_core == "rnn":
                cell = RNNCell(num_units=hidden_size, trainable=trainable)
            elif rnn_core == "lstm":
                cell = LSTMCell(num_units=hidden_size, trainable=trainable)
            elif rnn_core == "gru":
                cell = GRUCell(num_units=hidden_size, trainable=trainable)
            outputs, last_states = tf.nn.dynamic_rnn(
                inputs=embedding_output,
                cell=cell,
                dtype=tf.float64,
                sequence_length=seq_length)

        print(outputs)
        print(last_states)
