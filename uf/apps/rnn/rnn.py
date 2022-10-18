import math
import copy
import json
import random
import collections

from .._base_._base_ import BaseEncoder
from .. import util
from ...third import tf


class RNNEncoder(BaseEncoder):
    def __init__(
        self,
        is_training,
        input_ids,
        vocab_size,
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
            embedding_output, embedding_table = util.embedding_lookup(
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
            