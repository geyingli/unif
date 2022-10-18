""" Semantic-Parsing-Enhanced language modeling. """

import copy

from ...third import tf
from .._base_._base_ import BaseEncoder
from ..bert.bert import BERTEncoder
from .. import util


class SPEEncoder(BERTEncoder, BaseEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 position_ids,
                 scope="bert",
                 drop_pooler=False,
                 trainable=True,
                 **kwargs):

        bert_config = copy.deepcopy(bert_config)
        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        with tf.variable_scope(scope):
            with tf.variable_scope("embeddings"):

                (self.embedding_output, self.embedding_table) = \
                    self.embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=bert_config.vocab_size,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length,
                        embedding_size=bert_config.hidden_size,
                        initializer_range=bert_config.initializer_range,
                        word_embedding_name="word_embeddings",
                        tilda_embeddings=kwargs.get("tilda_embeddings"),
                        trainable=trainable)

                # Add positional embeddings and token type embeddings
                # layer normalize and perform dropout.
                self.embedding_output = self.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    position_ids=position_ids,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    hidden_size=bert_config.hidden_size,
                    use_token_type=True,
                    segment_ids=segment_ids,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=bert_config.initializer_range,
                    max_position_embeddings=\
                        bert_config.max_position_embeddings,
                    dropout_prob=bert_config.hidden_dropout_prob,
                    trainable=trainable)

            with tf.variable_scope("encoder"):
                attention_mask = self.create_attention_mask_from_input_mask(
                    input_mask, batch_size, max_seq_length)

                # stacked transformers
                self.all_encoder_layers = self.transformer_model(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    attention_mask=attention_mask,
                    hidden_size=bert_config.hidden_size,
                    num_hidden_layers=bert_config.num_hidden_layers,
                    num_attention_heads=bert_config.num_attention_heads,
                    intermediate_size=bert_config.intermediate_size,
                    intermediate_act_fn=util.get_activation(
                        bert_config.hidden_act),
                    hidden_dropout_prob=bert_config.hidden_dropout_prob,
                    attention_probs_dropout_prob=\
                    bert_config.attention_probs_dropout_prob,
                    initializer_range=bert_config.initializer_range,
                    trainable=trainable)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = self.sequence_output[:, 0, :]

                # trick: ignore the fully connected layer
                if drop_pooler:
                    self.pooled_output = first_token_tensor
                else:
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        bert_config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range),
                        trainable=trainable)

    def embedding_postprocessor(self,
                                input_tensor,
                                position_ids,
                                batch_size,
                                max_seq_length,
                                hidden_size,
                                use_token_type=False,
                                segment_ids=None,
                                token_type_vocab_size=16,
                                token_type_embedding_name=\
                                    "token_type_embeddings",
                                use_position_embeddings=True,
                                position_embedding_name="position_embeddings",
                                initializer_range=0.02,
                                max_position_embeddings=512,
                                dropout_prob=0.1,
                                dtype=tf.float32,
                                trainable=True):
        output = input_tensor

        if use_token_type:
            if segment_ids is None:
                raise ValueError(
                    "segment_ids must be specified if use_token_type is True.")
            token_type_table = tf.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)

            # This vocab will be small so we always do one-hot here,
            # since it is always faster for a small vocabulary.
            flat_segment_ids = tf.reshape(segment_ids, [-1])
            one_hot_ids = tf.one_hot(
                flat_segment_ids, depth=token_type_vocab_size, dtype=dtype)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(
                token_type_embeddings,
                [batch_size, max_seq_length, hidden_size])
            output += token_type_embeddings

        if use_position_embeddings:
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, hidden_size],
                initializer=util.create_initializer(initializer_range),
                dtype=dtype,
                trainable=trainable)
            output += tf.gather(full_position_embeddings, position_ids)

        output = util.layer_norm_and_dropout(
            output, dropout_prob, trainable=trainable)
        return output
