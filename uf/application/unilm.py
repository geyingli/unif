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
''' Applications based on UniLM. '''

import os
import random
import collections
import numpy as np

from ..tools import tf
from .base import LMModule
from .bert import BERTLM, create_instances_from_document
from ..modeling.unilm import UniLMEncoder
from ..modeling.bert import BERTDecoder, BERTConfig
from ..tokenization.word_piece import get_word_piece_tokenizer
from .. import utils


class UniLM(BERTLM, LMModule):
    ''' Language modeling on UniLM. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'init_checkpoint': (
            'A string that directs to the checkpoint file used for '
            'initialization')}

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_sample_next_sentence=True,
                 max_predictions_per_seq=20,
                 masked_lm_prob=0.15,
                 short_seq_prob=0.1,
                 do_whole_word_mask=False,
                 mode='bi',
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = 2
        self.do_sample_next_sentence = do_sample_next_sentence
        self.masked_lm_prob = masked_lm_prob
        self.short_seq_prob = short_seq_prob
        self.do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._max_predictions_per_seq = max_predictions_per_seq
        self.mode = mode
        self._id_to_label = None
        self.__init_args__ = locals()

        assert mode in ('bi', 'l2r', 'r2l', 's2s'), (
            'Wrong value of `mode`: %s. Pick one from `bi` (bidirectional), '
            '`l2r` (left-to-right), `r2l` (right-to-left) and '
            '`s2s` (seq-to-seq).' % mode)
        tf.logging.info(
            'LM Mode: `%s`. Use method `.to_mode()` to convert it into '
            '`bi`, `l2r`, `r2l` or `s2s`.' % mode)
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
        if '[EOS]' not in self.tokenizer.vocab:
            self.tokenizer.add('[EOS]')
            self.bert_config.vocab_size += 1
            tf.logging.info('Add necessary token `[EOS]` into vocabulary.')

    def to_mode(self, mode):
        ''' Switch the mode of UniLM.

        Args:
            mode: string. One of `bi` (bidirectional), `l2r` (left-to-right),
              `r2l` (right-to-left) and `s2s` (seq-to-seq).
        Returns:
            None
        '''
        assert mode in ('bi', 'l2r', 'r2l', 's2s'), (
            'Wrong value of `mode`: %s. Pick one from `bi` (bidirectional), '
            '`l2r` (left-to-right), `r2l` (right-to-left) and '
            '`s2s` (seq-to-seq).' % mode)

        if mode != self.mode:
            self._session_mode = None
        self.mode = mode

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            if self.mode == 'bi':
                if y is not None:
                    assert not self.do_sample_next_sentence, (
                        '`y` should be None when `do_sample_next_sentence` '
                        'is True.')
                else:
                    assert self.do_sample_next_sentence, (
                        '`y` can\'t be None when `do_sample_next_sentence` '
                        'is False.')
            else:
                assert y is None, (
                    'Only training of bidirectional LM is supervised. '
                    '`y` should be None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights,
             next_sentence_labels) = self._convert_X(
                 X_tokenized if tokenized else X,
                 is_training, tokenized=tokenized)

            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            data['masked_lm_positions'] = \
                np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data['masked_lm_ids'] = \
                    np.array(masked_lm_ids, dtype=np.int32)
                data['masked_lm_weights'] = \
                    np.array(masked_lm_weights, dtype=np.float32)
                data['next_sentence_labels'] = \
                    np.array(next_sentence_labels, dtype=np.int32)

            n_inputs = len(input_ids)
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
            if self.mode in ('l2r', 'r2l'):
                info = '`l2r` or `r2l` only supports single sentence inputs.'
                if not tokenized:
                    assert isinstance(example, str), info
                else:
                    assert isinstance(example[0], str), info
            elif self.mode == 's2s':
                info = '`s2s` only supports 2-sentence inputs.'
                assert len(example) == 2, info
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                raise ValueError(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        input_ids = []
        input_mask = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_labels = []

        # random sampling of next sentence
        if is_training and self.mode == 'bi' and self.do_sample_next_sentence:
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
        else:
            next_sentence_labels = [1] * len(segment_input_tokens)

        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _masked_lm_positions = []
            _masked_lm_ids = []
            _masked_lm_weights = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ['[SEP]'])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            # special values for `_input_tokens` and `input_mask`
            if self.mode == 's2s':
                _input_tokens.pop()
                _input_tokens.append('[EOS]')
                _input_mask = [len(_input_ids)] * (len(segments[0]) + 2)
                for i in range(len(segments[1]) + 1):
                    _input_mask.append(_input_mask[0] + i + 1)

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
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids, input_mask, segment_ids,
                masked_lm_positions, masked_lm_ids, masked_lm_weights,
                next_sentence_labels)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = UniLMEncoder(
            mode=self.mode,
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
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
               self._preds['MLM_preds'], self._losses['MLM_losses']]
        if self.mode == 'bi':
            ops += [self._preds['NSP_preds'], self._losses['NSP_losses']]
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
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = (
            np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) /
            batch_mlm_mask.sum())

        # MLM loss
        batch_mlm_losses = output_arrays[2]
        mlm_loss = np.mean(batch_mlm_losses)

        info = ''
        info += ', MLM accuracy %.4f' % mlm_accuracy
        info += ', MLM loss %.6f' % mlm_loss

        if self.mode == 'bi':

            # NSP accuracy
            batch_nsp_preds = output_arrays[3]
            nsp_accuracy = np.mean(batch_nsp_preds == batch_nsp_labels)

            # NSP loss
            batch_nsp_losses = output_arrays[4]
            nsp_loss = np.mean(batch_nsp_losses)

            info += ', NSP accuracy %.4f' % nsp_accuracy
            info += ', NSP loss %.6f' % nsp_loss

        return info

    def _get_predict_ops(self):
        ops = [self._preds['MLM_preds']]
        if self.mode == 'bi':
            ops += [self._preds['NSP_preds'], self._probs['NSP_probs']]
        return ops

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # MLM preds
        mlm_preds = []
        mlm_positions = self.data['masked_lm_positions']
        all_preds = utils.transform(output_arrays[0], n_inputs)
        for ex_id, _preds in enumerate(all_preds):
            _ids = []
            for p_id, _id in enumerate(_preds):
                if mlm_positions[ex_id][p_id] == 0:
                    break
                _ids.append(_id)
            mlm_preds.append(self.tokenizer.convert_ids_to_tokens(_ids))

        outputs = {}
        outputs['mlm_preds'] = mlm_preds

        if self.mode == 'bi':

            # NSP preds
            nsp_preds = utils.transform(output_arrays[1], n_inputs).tolist()

            # NSP probs
            nsp_probs = utils.transform(output_arrays[2], n_inputs)

            outputs['nsp_preds'] = nsp_preds
            outputs['nsp_probs'] = nsp_probs

        return outputs


MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])


def create_masked_lm_predictions(tokens,
                                 masked_lm_prob,
                                 max_predictions_per_seq,
                                 vocab_words,
                                 ngram=3,
                                 do_whole_word_mask=True):
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
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith('##')):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

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

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                             np.sum(pvals[:len(cand_index_set)]))
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
                    masked_token = vocab_words[
                        random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_bert_config(config_file=None):
    if not os.path.exists(config_file):
        raise ValueError(
            'Can\'t find config_file \'%s\'. '
            'Please pass the correct path of configuration file, '
            'e.g.`bert_config.json`.' % config_file)
    return BERTConfig.from_json_file(config_file)


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 2,
        '/pooler/': 1,
        'cls/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/layer_%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
