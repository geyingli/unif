# coding:=utf-8
# Copyright 2021 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Applications based on semantic parsing enhanced language modeling. '''

import random
import collections
import numpy as np

from uf.tools import tf
from .base import LMModule
from .bert import (BERTLM, get_bert_config,
                   get_word_piece_tokenizer, get_key_to_depths,
                   create_instances_from_document,
                   create_masked_lm_predictions)
from uf.modeling.bert import BERTEncoder, BERTDecoder
import uf.utils as utils




class SPELM(BERTLM, LMModule):
    ''' Language modeling on SPE. '''
    _INFER_ATTRIBUTES = BERTLM._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 max_predictions_per_seq=20,
                 masked_lm_prob=0.15,
                 do_whole_word_mask=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(BERTLM, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = masked_lm_prob
        self.do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._max_predictions_per_seq = max_predictions_per_seq
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

        if '[CLS]' not in self.tokenizer.vocab:
            self.tokenizer.add('[CLS]')
            self.bert_config.vocab_size += 1
            tf.logging.info('Add necessary token `[CLS]` into vocabulary.')
        if '[SEP]' not in self.tokenizer.vocab:
            self.tokenizer.add('[SEP]')
            self.bert_config.vocab_size += 1
            tf.logging.info('Add necessary token `[SEP]` into vocabulary.')

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, (
            'Training of %s is unsupervised. `y` should be None.'
            % self.__class__.__name__)

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized

            (left_input_ids, right_input_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights,
             next_sentence_labels) = self._convert_X(
                 X_tokenized if tokenized else X,
                 is_training, tokenized=tokenized)

            data['left_input_ids'] = np.array(left_input_ids, dtype=np.int32)
            data['right_input_ids'] = np.array(right_input_ids, dtype=np.int32)
            data['masked_lm_positions'] = \
                np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data['masked_lm_ids'] = \
                    np.array(masked_lm_ids, dtype=np.int32)
                data['masked_lm_weights'] = \
                    np.array(masked_lm_weights, dtype=np.float32)
                data['next_sentence_labels'] = \
                    np.array(next_sentence_labels, dtype=np.int32)

            n_inputs = len(left_input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            next_sentence_labels = self._convert_y(y)
            data['next_sentence_labels'] = \
                np.array(next_sentence_labels, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        input_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_labels = []

        # random sampling of next sentence
        if is_training:
            new_segment_input_tokens = []
            for ex_id in range(len(segment_input_tokens)):
                instances = create_instances_from_document(
                    all_documents=segment_input_tokens,
                    document_index=ex_id,
                    max_seq_length=self.max_seq_length - 3,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    short_seq_prob=self.short_seq_prob,
                    vocab_words=list(self.tokenizer.vocab.keys()))
                for (segments, is_random_next) in instances:
                    new_segment_input_tokens.append(segments)
                    next_sentence_labels.append(is_random_next)
            segment_input_tokens = new_segment_input_tokens

        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _masked_lm_positions = []
            _masked_lm_ids = []
            _masked_lm_weights = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _input_tokens.extend(segment + ['[SEP]'])

            # random sampling of masked tokens
            if is_training:
                if (ex_id + 1) % 10000 == 0:
                    tf.logging.info(
                        'Sampling masks of input %d' % (ex_id + 1))
                (_input_tokens, _masked_lm_positions, _masked_lm_labels) = \
                    create_masked_lm_predictions(
                        tokens=_input_tokens,
                        masked_lm_prob=self.masked_lm_prob,
                        max_predictions_per_seq=self._max_predictions_per_seq,
                        vocab_words=list(self.tokenizer.vocab.keys()),
                        do_whole_word_mask=self.do_whole_word_mask)
                _masked_lm_ids = \
                    self.tokenizer.convert_tokens_to_ids(_masked_lm_labels)
                _masked_lm_weights = [1.0] * len(_masked_lm_positions)

                # padding
                for _ in range(self._max_predictions_per_seq -
                               len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)
                    _masked_lm_ids.append(0)
                    _masked_lm_weights.append(0.0)
            else:
                # `masked_lm_positions` is required for both training
                # and inference of BERT language modeling.
                for i in range(len(_input_tokens)):
                    if _input_tokens[i] == '[MASK]':
                        _masked_lm_positions.append(i)

                # padding
                for _ in range(self._max_predictions_per_seq -
                               len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)

            input_ids.append(_input_ids)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids,
                masked_lm_positions, masked_lm_ids, masked_lm_weights,
                next_sentence_labels)

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'masked_lm_positions': utils.get_placeholder(
                target, 'masked_lm_positions',
                [None, self._max_predictions_per_seq], tf.int32),
            'masked_lm_ids': utils.get_placeholder(
                target, 'masked_lm_ids',
                [None, self._max_predictions_per_seq], tf.int32),
            'masked_lm_weights': utils.get_placeholder(
                target, 'masked_lm_weights',
                [None, self._max_predictions_per_seq], tf.float32),
            'next_sentence_labels': utils.get_placeholder(
                target, 'next_sentence_labels',
                [None], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
            **kwargs)
        decoder = BERTDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            encoder=encoder,
            masked_lm_positions=split_placeholders['masked_lm_positions'],
            masked_lm_ids=split_placeholders['masked_lm_ids'],
            masked_lm_weights=split_placeholders['masked_lm_weights'],
            next_sentence_labels=split_placeholders['next_sentence_labels'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope_lm='cls/predictions',
            scope_cls='cls/seq_relationship',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['MLM_preds'], self._preds['NSP_preds'],
               self._losses['MLM_losses'], self._losses['NSP_losses']]
        if as_feature:
            ops.extend(
                [self.placeholders['masked_lm_positions'],
                 self.placeholders['masked_lm_ids'],
                 self.placeholders['next_sentence_labels']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mlm_positions = output_arrays[-3]
            batch_mlm_labels = output_arrays[-2]
            batch_nsp_labels = output_arrays[-1]
        else:
            batch_mlm_positions = \
                feed_dict[self.placeholders['masked_lm_positions']]
            batch_mlm_labels = \
                feed_dict[self.placeholders['masked_lm_ids']]
            batch_nsp_labels = \
                feed_dict[self.placeholders['next_sentence_labels']]

        # MLM accuracy
        batch_mlm_preds = output_arrays[1]
        batch_mlm_positions = np.reshape(batch_mlm_positions, [-1])
        batch_mlm_labels = np.reshape(batch_mlm_labels, [-1])
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = (
            np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) /
            batch_mlm_mask.sum())

        # NSP accuracy
        batch_nsp_preds = output_arrays[2]
        nsp_accuracy = np.mean(batch_nsp_preds == batch_nsp_labels)

        # MLM loss
        batch_mlm_losses = output_arrays[3]
        mlm_loss = np.mean(batch_mlm_losses)

        # NSP loss
        batch_nsp_losses = output_arrays[4]
        nsp_loss = np.mean(batch_nsp_losses)

        info = ''
        info += ', MLM accuracy %.4f' % mlm_accuracy
        info += ', NSP accuracy %.4f' % nsp_accuracy
        info += ', MLM loss %.6f' % mlm_loss
        info += ', NSP loss %.6f' % nsp_loss

        return info

    def _get_predict_ops(self):
        return [self._preds['MLM_preds'],
                self._preds['NSP_preds'],
                self._probs['NSP_probs']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # MLM preds
        mlm_preds = []
        mlm_positions = self.data['masked_lm_positions']
        all_preds = utils.transform(output_arrays[0], n_inputs, reshape=True)
        for ex_id, _preds in enumerate(all_preds):
            _ids = []
            for p_id, _id in enumerate(_preds):
                if mlm_positions[ex_id][p_id] == 0:
                    break
                _ids.append(_id)
            mlm_preds.append(self.tokenizer.convert_ids_to_tokens(_ids))

        # NSP preds
        nsp_preds = utils.transform(output_arrays[1], n_inputs).tolist()

        # NSP probs
        nsp_probs = utils.transform(output_arrays[2], n_inputs)

        outputs = {}
        outputs['mlm_preds'] = mlm_preds
        outputs['nsp_preds'] = nsp_preds
        outputs['nsp_probs'] = nsp_probs

        return outputs


def create_instances_from_document(all_documents, document_index,
                                   max_seq_length, masked_lm_prob,
                                   max_predictions_per_seq,
                                   vocab_words):
    document = all_documents[document_index]

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments 'A' and 'B' based on the actual 'sentences' provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= max_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into
                # the `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = max_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for
                    # large corpora. However, just to be careful, we try to
                    # make sure that the random document is not the same as
                    # the document we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(
                            0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we 'put them
                    # back' so they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instances.append(([tokens_a, tokens_b], is_random_next))
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])


def create_masked_lm_predictions(tokens,
                                 masked_lm_prob,
                                 max_predictions_per_seq,
                                 vocab_words,
                                 do_whole_word_mask=False):
    ''' Creates the predictions for the masked LM objective. '''

    cand_indexes = []
    for (i, token) in enumerate(tokens):
      if token == '[CLS]' or token == '[SEP]':
        continue
      # Whole Word Masking means that if we mask all of the wordpieces
      # corresponding to an original word. When a word has been split into
      # WordPieces, the first token does not have any marker and any
      # subsequence tokens are prefixed with ##. So whenever we see the
      # `##` token, we append it to the previous set of word indexes.
      #
      # Note that Whole Word Masking does *not* change the training code
      # at all -- we still predict each WordPiece independently, softmaxed
      # over the entire vocabulary.
      if (do_whole_word_mask and len(cand_indexes) >= 1 and
          token.startswith('##')):
        cand_indexes[-1].append(i)
      else:
        cand_indexes.append([i])

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
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
                masked_token = '[MASK]'
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[
                        random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)
