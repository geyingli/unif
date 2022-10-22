""" Revised from Transformer, especially for chatbot. """

from ...third import tf
from ..transformer.transformer import *
from .._base_._base_ import BaseEncoder, BaseDecoder
from .. import util


class Chatbot(BaseDecoder, BaseEncoder):
    def __init__(self,
                 vocab_size,
                 is_training,
                 source_ids,
                 target_ids,
                 sos_id,
                 sample_weight=None,
                 hidden_size=768,
                 num_blocks=6,
                 num_attention_heads=12,
                 scope="transformer",
                 use_label_smoothing=False,
                 **kwargs):
        super().__init__()

        dropout_rate = 0.0
        if is_training:
            dropout_rate = 0.1

        source_shape = util.get_shape_list(source_ids, expected_rank=2)
        target_shape = util.get_shape_list(target_ids, expected_rank=2)
        batch_size = source_shape[0]
        source_max_seq_length = source_shape[1]
        target_max_seq_length = target_shape[1]

        with tf.variable_scope(scope):
            source_mask = tf.math.equal(source_ids, 0)

            # embedding
            with tf.variable_scope("embeddings"):
                enc, embedding_table = embedding_lookup(
                    input_ids=source_ids,
                    vocab_size=vocab_size,
                    batch_size=batch_size,
                    max_seq_length=source_max_seq_length,
                    embedding_size=hidden_size,
                    word_embedding_name="word_embeddings",
                    tilda_embeddings=kwargs.get("tilda_embeddings"),
                )
                enc *= hidden_size ** 0.5  # scale
                enc += positional_encoding(enc, source_max_seq_length)
                enc = util.dropout(enc, dropout_rate)

            with tf.variable_scope("encoder"):

                # stacked multi-attention layers
                for i in range(num_blocks):
                    with tf.variable_scope("block_%s" % i):

                        # self-attention
                        enc = multihead_attention(
                            queries=enc,
                            keys=enc,
                            values=enc,
                            key_masks=source_mask,
                            num_heads=num_attention_heads,
                            dropout_rate=dropout_rate,
                            training=is_training,
                            causality=False,
                            scope="self_attention",
                        )

                        # feed forward
                        enc = ff(enc, num_units=[hidden_size * 4, hidden_size])
                memory = enc

            def _forward(target_ids, target_mask, target_max_seq_length):

                with tf.variable_scope("decoder"):

                    # shared embedding
                    dec = tf.nn.embedding_lookup(embedding_table, target_ids)
                    dec *= hidden_size ** 0.5  # scale
                    dec += positional_encoding(dec, target_max_seq_length)
                    dec = util.dropout(dec, dropout_rate)

                    # blocks
                    for i in range(num_blocks):
                        with tf.variable_scope("block_%s" % i):

                            # masked self-attention
                            dec = multihead_attention(
                                queries=dec,
                                keys=dec,
                                values=dec,
                                key_masks=target_mask,
                                num_heads=num_attention_heads,
                                dropout_rate=dropout_rate,
                                training=is_training,
                                causality=True,
                                scope="masked_self_attention",
                            )

                            # vanilla attention
                            dec = multihead_attention(
                                queries=dec,
                                keys=memory,
                                values=memory,
                                key_masks=source_mask,
                                num_heads=num_attention_heads,
                                dropout_rate=dropout_rate,
                                training=is_training,
                                causality=False,
                                scope="vanilla_attention",
                            )

                            # feed forward
                            dec = ff(dec, num_units=[4 * hidden_size, hidden_size])

                # final linear projection (embedding weights are shared)
                with tf.variable_scope("cls"):
                    output_bias = tf.get_variable(
                        "output_bias", 
                        shape=[vocab_size],
                        initializer=tf.zeros_initializer(),
                    )
                    dec = tf.reshape(dec, [-1, hidden_size])
                    logits = tf.matmul(dec, embedding_table, transpose_b=True)
                    logits = tf.reshape(logits, [-1, target_max_seq_length, vocab_size])

                    # Accelerate training by eliminating bias with transition 
                    # probabilities in real world.
                    self.transition_matrix = tf.get_variable(
                        "transition_matrix",
                        shape=[vocab_size, vocab_size],
                        initializer=util.create_initializer(),
                        dtype=tf.float32,
                        trainable=False,
                    )
                    transition_probs, _ = util.embedding_lookup(
                        target_ids,
                        vocab_size=vocab_size,
                        batch_size=batch_size,
                        max_seq_length=target_max_seq_length,
                        embeddings=self.transition_matrix,
                        embedding_size=vocab_size,
                    )
                    logits *= transition_probs

                    logits = tf.nn.bias_add(logits, output_bias)

                return logits

            # forward once: training
            target_mask = tf.math.equal(target_ids, 0)  # (N, T2)
            logits = _forward(target_ids, target_mask, target_max_seq_length)

            # convert to labels
            label_ids = tf.concat(
                [target_ids[:, 1:], tf.zeros([batch_size, 1], dtype=tf.int32)], 
                axis=-1,
            )

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids, depth=vocab_size)
            if use_label_smoothing:
                one_hot_labels = label_smoothing(one_hot_labels)
            per_token_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            label_mask = tf.cast(tf.not_equal(label_ids, 0), tf.float32)
            per_example_loss = \
                tf.reduce_sum(per_token_loss * label_mask, axis=-1) / \
                tf.reduce_sum(label_mask, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            self.train_loss = tf.reduce_mean(per_example_loss)
            self.tensors["losses"] = per_example_loss

            # forward loop: prediction
            target_mask_base = tf.zeros([batch_size, 1], dtype=tf.int32)
            target_ids = tf.ones([batch_size, 1], dtype=tf.int32) * sos_id

            for cur_length in range(1, target_max_seq_length + 1):
                target_mask = tf.tile(target_mask_base, [1, cur_length])
                logits = _forward(target_ids, target_mask, cur_length)

                pred_ids = tf.argmax(
                    logits[:, cur_length-1:cur_length, :],
                    axis=-1)
                pred_ids = tf.cast(pred_ids, tf.int32)
                target_ids = tf.concat([target_ids, pred_ids], axis=-1)

            self.tensors["preds"] = target_ids[:, 1:]
