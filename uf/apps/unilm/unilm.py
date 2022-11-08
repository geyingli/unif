""" UniLM, a unified model of bidirectional modeling, unidirectional modeling and sequence-to-sequence modeling. """

import random
import collections
import numpy as np

from .._base_._base_ import BaseEncoder
from ..bert.bert import BERTEncoder
from ...third import tf


class UniLMEncoder(BERTEncoder, BaseEncoder):
    def __init__(self,
                 mode,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 prompt_length=None,
                 scope="bert",
                 drop_pooler=False,
                 trainable=True,
                 **kwargs):
        self._mode = mode
        super().__init__(
            bert_config,
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            scope,
            drop_pooler,
            trainable,
            **kwargs)
        self.prompt_length = prompt_length

    def create_attention_mask_from_input_mask(self,
                                              input_mask,
                                              batch_size,
                                              max_seq_length,
                                              dtype=tf.float32):
        if self._mode == "bi":
            to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, max_seq_length]), dtype=dtype)
            broadcast_ones = tf.ones(shape=[batch_size, max_seq_length, 1], dtype=dtype)
            mask = broadcast_ones * to_mask

        elif self._mode == "l2r":
            arange = tf.range(max_seq_length) + 1
            to_mask = tf.cast(tf.sequence_mask(arange, max_seq_length), dtype)
            to_mask = tf.reshape(to_mask, [1, max_seq_length, max_seq_length])
            mask = tf.tile(to_mask, [batch_size, 1, 1])

        elif self._mode == "r2l":
            to_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, max_seq_length]), dtype=dtype)
            broadcast_ones = tf.ones(shape=[batch_size, max_seq_length, 1], dtype=dtype)
            cover_mask = broadcast_ones * to_mask

            if self.prompt_length is not None:
                prompt = tf.tile(tf.reshape(self.prompt_length, [batch_size, 1]), [1, max_seq_length])
                prompt_mask = tf.cast(tf.sequence_mask(prompt, max_seq_length), dtype)
                reverse_mask = tf.cast(tf.sequence_mask(input_mask, max_seq_length), dtype)
                mask = (1 - reverse_mask + prompt_mask) * cover_mask
            else:
                arange = tf.range(max_seq_length)
                reverse = tf.cast(tf.sequence_mask(arange, max_seq_length), dtype)
                reverse = tf.reshape(reverse, [1, max_seq_length, max_seq_length])
                reverse_mask = tf.tile(reverse, [batch_size, 1, 1])
                mask = (1 - reverse_mask) * cover_mask

        elif self._mode == "s2s":
            mask = tf.cast(tf.sequence_mask(input_mask, max_seq_length), dtype)

        return mask


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, ngram=3, do_whole_word_mask=True):
    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = [0.8, 0.1, 0.1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or
        # previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(
            ngrams[:len(cand_index_set)],
            p=pvals[:len(cand_index_set)] / np.sum(pvals[:len(cand_index_set)]),
        )
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": num_hidden_layers + 2,
        "/pooler/": 1,
        "cls/": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/layer_%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power

