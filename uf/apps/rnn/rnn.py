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
        use_tilda_embedding=False,
        **kwargs,
    ):
        
        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        if use_tilda_embedding:
            with tf.variable_scope("", reuse=True):
                tilda_embeddings = tf.get_variable("tilda_embeddings")

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        with tf.variable_scope(scope):
            embedding_output, embedding_table = self.embedding_lookup(
                input_ids=input_ids,
                vocab_size=vocab_size,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                embedding_size=hidden_size,
                initializer_range=initializer_range,
                word_embedding_name="word_embeddings",
                tilda_embeddings=tilda_embeddings,
                trainable=trainable,
            )

    def embedding_lookup(
        self,
        input_ids,
        vocab_size,
        batch_size,
        max_seq_length,
        embedding_size=128,
        initializer_range=0.02,
        word_embedding_name="word_embeddings",
        dtype=tf.float32,
        trainable=True,
        tilda_embeddings=None,
    ):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        if tilda_embeddings is not None:
            embedding_table = tilda_embeddings
        else:
            embedding_table = tf.get_variable(
                name=word_embedding_name,
                shape=[vocab_size, embedding_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable,
            )

        flat_input_ids = tf.reshape(input_ids, [-1])
        output = tf.gather(embedding_table, flat_input_ids, name="embedding_look_up")
        output = tf.reshape(output, [batch_size, max_seq_length, embedding_size])
        return (output, embedding_table)