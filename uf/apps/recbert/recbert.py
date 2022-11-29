""" Recorrection language modeling. """

import random
import numpy as np

from ...third import tf
from .._base_._base_ import BaseDecoder
from .._base_._base_classifier import ClsDecoder
from ..bert.bert import BERTEncoder
from .. import util


class RecBERT(BaseDecoder, BERTEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 add_label_ids,
                 del_label_ids,
                 cls_label_ids,
                 sample_weight=None,
                 add_prob=0,
                 del_prob=0,
                 scope="bert",
                 **kwargs):
        super().__init__()

        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.float32)

        shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = shape[0]
        max_seq_length = shape[1]

        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        with tf.variable_scope(scope):

            # forward once
            hidden = self._bert_forward(
                bert_config,
                input_ids,
                input_mask,
                batch_size,
                max_seq_length,
                tilda_embeddings=kwargs.get("tilda_embeddings"))

            self.train_loss = 0
            self._lm_forward(
                is_training,
                input_tensor=hidden,
                input_mask=input_mask,
                label_ids=add_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=add_prob,
                scope="add",
                name="add",
                sample_weight=sample_weight,
                **kwargs,
            )
            self._seq_cls_forward(
                is_training,
                input_tensor=hidden,
                input_mask=input_mask,
                label_ids=del_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                prob=del_prob,
                scope="del",
                name="del",
                sample_weight=sample_weight,
                **kwargs,
            )
            self._cls_forward(
                is_training,
                input_tensor=hidden[:, 0, :],
                label_ids=cls_label_ids,
                bert_config=bert_config,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                scope="cls",
                name="cls",
                sample_weight=sample_weight,
                **kwargs,
            )

    def _bert_forward(self,
                     bert_config,
                     input_ids,
                     input_mask,
                     batch_size,
                     max_seq_length,
                     dtype=tf.float32,
                     trainable=True,
                     tilda_embeddings=None):

        with tf.variable_scope("embeddings"):

            (embedding_output, self.embedding_table) = util.embedding_lookup(
                input_ids=input_ids,
                vocab_size=bert_config.vocab_size,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                embeddings=tilda_embeddings,
                embedding_size=bert_config.hidden_size,
                initializer_range=bert_config.initializer_range,
                word_embedding_name="word_embeddings",
                dtype=dtype,
                trainable=trainable)

            # Add positional embeddings and token type embeddings
            # layer normalize and perform dropout.
            embedding_output = self.embedding_postprocessor(
                input_tensor=embedding_output,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                hidden_size=bert_config.hidden_size,
                use_token_type=False,
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=bert_config.initializer_range,
                max_position_embeddings=\
                    bert_config.max_position_embeddings,
                dropout_prob=bert_config.hidden_dropout_prob,
                dtype=dtype,
                trainable=trainable)

        with tf.variable_scope("encoder"):
            attention_mask = self.create_attention_mask_from_input_mask(
                input_mask, batch_size, max_seq_length, dtype=dtype)

            # stacked transformers
            all_encoder_layers = self.transformer_model(
                input_tensor=embedding_output,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                attention_mask=attention_mask,
                hidden_size=bert_config.hidden_size,
                num_hidden_layers=bert_config.num_hidden_layers,
                num_attention_heads=bert_config.num_attention_heads,
                intermediate_size=bert_config.intermediate_size,
                intermediate_act_fn=util.get_activation(bert_config.hidden_act),
                hidden_dropout_prob=bert_config.hidden_dropout_prob,
                attention_probs_dropout_prob=\
                bert_config.attention_probs_dropout_prob,
                initializer_range=bert_config.initializer_range,
                dtype=dtype,
                trainable=trainable)

        return all_encoder_layers[-1]

    def _lm_forward(self,
                    is_training,
                    input_tensor,
                    input_mask,
                    label_ids,
                    bert_config,
                    batch_size,
                    max_seq_length,
                    prob,
                    scope,
                    name,
                    sample_weight=None,
                    **kwargs):

        with tf.variable_scope(scope):

            with tf.variable_scope("verifier"):
                logits = tf.layers.dense(
                    input_tensor, 2,
                    kernel_initializer=util.create_initializer(bert_config.initializer_range),
                    trainable=True,
                )
                verifier_label_ids = tf.cast(tf.greater(label_ids, 0), tf.int32)

                # loss
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                one_hot_labels = tf.one_hot(verifier_label_ids, depth=2)
                per_token_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

                input_mask = tf.cast(input_mask, tf.float32)
                per_token_loss *= input_mask / tf.reduce_sum(input_mask, keepdims=True, axis=-1)
                per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
                if sample_weight is not None:
                    per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

                if prob != 0:
                    self.train_loss += tf.reduce_mean(per_example_loss)
                verifier_loss = per_example_loss
                verifier_preds = tf.argmax(logits, axis=-1)

            with tf.variable_scope("prediction"):

                with tf.variable_scope("intermediate"):
                    logits = tf.layers.dense(
                        input_tensor, bert_config.hidden_size * 4,
                        kernel_initializer=util.create_initializer(bert_config.initializer_range),
                        activation=util.gelu,
                        trainable=True,
                    )
                with tf.variable_scope("output"):
                    logits = tf.layers.dense(
                        logits, bert_config.hidden_size,
                        kernel_initializer=util.create_initializer(bert_config.initializer_range),
                        trainable=True,
                    )

                flattened = tf.reshape(logits, [batch_size * max_seq_length, bert_config.hidden_size])
                logits = tf.matmul(flattened, self.embedding_table, transpose_b=True)
                logits = tf.reshape(logits, [-1, max_seq_length, bert_config.vocab_size])

                # loss
                per_token_loss = util.cross_entropy(logits, label_ids, bert_config.vocab_size, **kwargs)
                input_mask *= tf.cast(verifier_preds, tf.float32)
                per_token_loss *= input_mask / (tf.reduce_sum(input_mask, keepdims=True, axis=-1) + 1e-6)
                per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
                if sample_weight is not None:
                    per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

                if prob != 0:
                    self.train_loss += tf.reduce_mean(per_example_loss)
                self.tensors[name + "_loss"] = verifier_loss
                self.tensors[name + "_preds"] = tf.argmax(logits, axis=-1) * verifier_preds

    def _seq_cls_forward(self,
                     is_training,
                     input_tensor,
                     input_mask,
                     label_ids,
                     bert_config,
                     batch_size,
                     max_seq_length,
                     prob,
                     scope,
                     name,
                     sample_weight=None,
                     **kwargs):

        with tf.variable_scope(scope):
            logits = tf.layers.dense(
                input_tensor, 2,
                kernel_initializer=util.create_initializer(bert_config.initializer_range),
                trainable=True)

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids, depth=2)
            per_token_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            if prob != 0:
                self.train_loss += tf.reduce_mean(per_example_loss)
            self.tensors[name + "_loss"] = per_example_loss
            self.tensors[name + "_preds"] = tf.argmax(logits, axis=-1)

    def _cls_forward(self,
                     is_training,
                     input_tensor,
                     label_ids,
                     bert_config,
                     batch_size,
                     max_seq_length,
                     scope,
                     name,
                     sample_weight=None,
                     **kwargs):

        if kwargs.get("is_logits"):
            logits = input_tensor
        else:
            if kwargs.get("return_hidden"):
                self.tensors["hidden"] = input_tensor

            hidden_size = input_tensor.shape.as_list()[-1]
            with tf.variable_scope(scope):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[2, hidden_size],
                    initializer=util.create_initializer(bert_config.initializer_range),
                    trainable=True,
                )
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[2],
                    initializer=tf.zeros_initializer(),
                    trainable=True,
                )
                output_layer = util.dropout(input_tensor, bert_config.hidden_dropout_prob if is_training else 0.0)
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

        self.tensors[name + "_preds"] = tf.argmax(logits, axis=-1, name="preds")
        self.tensors[name + "_probs"] = tf.nn.softmax(logits, axis=-1, name="probs")

        per_example_loss = util.cross_entropy(logits, label_ids, 2, **kwargs)
        if sample_weight is not None:
            per_example_loss *= sample_weight
        self.tensors[name + "_loss"] = per_example_loss
        self.train_loss += tf.reduce_mean(per_example_loss)


def sample_wrong_tokens(_input_ids, _add_label_ids, _del_label_ids, max_add, max_del, nonpad_seq_length, vocab_size, vocab_ind, vocab_p):

    # `add`, remove padding for prediction of adding tokens
    # e.g. input_ids: 124 591 9521 -> 124 9521
    #      add_label_ids: 0 0 0 -> 591 0
    for _ in range(max_add):
        cand_indicies = [
            i for i in range(0, len(_input_ids) - 2)
            if _input_ids[i + 1] != 0 and _add_label_ids[i] == 0 and _add_label_ids[i + 1] == 0
        ]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _id = _input_ids[index + 1]

        _input_ids.pop(index + 1)
        _add_label_ids.pop(index + 1)
        _del_label_ids.pop(index + 1)

        _add_label_ids[index] = _id

        _input_ids.append(0)
        _add_label_ids.append(0)
        _del_label_ids.append(0)
        nonpad_seq_length -= 1

    # `del`, add wrong tokens for prediction of deleted tokens
    # e.g. input_ids: 124 591 -> 92 124 591
    #      del_label_ids: 0 0 -> 1 0 0
    for _ in range(max_del):
        if _input_ids[-1] != 0:  # no more space
            break

        index = random.randint(1, nonpad_seq_length - 1)
        _id = np.random.choice(vocab_ind, p=vocab_p)  # sample from distribution of vocabulary

        _input_ids.insert(index, _id)
        _add_label_ids.insert(index, 0)
        _del_label_ids.insert(index, 1)

        # when new token is the same with the right one, only delete the right
        if _id == _input_ids[index + 1]:
            _del_label_ids[index + 1], _del_label_ids[index] = _del_label_ids[index], _del_label_ids[index + 1]

        _input_ids.pop()
        _add_label_ids.pop()
        _del_label_ids.pop()

        nonpad_seq_length += 1


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": num_hidden_layers + 2,
        "/pooler/": 1,
        "add/": 0,
        "del/": 0,
        "cls/": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/layer_%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power
