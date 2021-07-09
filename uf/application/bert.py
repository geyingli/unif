# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
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
''' Applications based on BERT. '''

import os
import random
import collections
import numpy as np

from uf.tools import tf
from .base import ClassifierModule, LMModule, NERModule, MRCModule
from uf.modeling.bert import BERTEncoder, BERTDecoder, BERTConfig
from uf.modeling.base import (
    CLSDecoder, BinaryCLSDecoder, SeqCLSDecoder, MRCDecoder)
from uf.modeling.crf import CRFDecoder, viterbi_decode
from uf.tokenization.word_piece import get_word_piece_tokenizer
import uf.utils as utils



class BERTClassifier(ClassifierModule):
    ''' Single-label classifier on BERT. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'label_size': (
            'An integer that defines number of possible labels of outputs'),
        'init_checkpoint': (
            'A string that directs to the checkpoint file used for '
            'initialization')}

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
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
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'
        if is_parallel:
            assert self.label_size, ('Can\'t parse data on multi-processing '
                'when `label_size` is None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
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
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ['[SEP]'])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _convert_x(self, x, tokenized):
        if not tokenized:
            # deal with general inputs
            if isinstance(x, str):
                return [self.tokenizer.tokenize(x)]

            # deal with multiple inputs
            return [self.tokenizer.tokenize(seg) for seg in x]

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return [x]

        # deal with tokenized and multiple inputs
        return x

    def _convert_y(self, y):
        label_set = set(y)

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, (
                'Number of unique `y`s exceeds `label_size`.')
        else:
            self.label_size = len(label_set)

        # automatically set `id_to_label`
        if not self._id_to_label:
            self._id_to_label = list(label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._id_to_label = list(sorted(self._id_to_label))
            except Exception:
                pass
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {
            label: index for index, label in enumerate(self._id_to_label)}

        label_ids = [self._label_to_id[label] for label in y]
        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None], tf.int32),
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
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = CLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/seq_relationship',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['preds'], self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # accuracy
        batch_preds = output_arrays[1]
        accuracy = np.mean(batch_preds == batch_labels)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', accuracy %.4f' % accuracy
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['probs']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = utils.transform(output_arrays[0], n_inputs)

        # preds
        preds = np.argmax(probs, axis=-1).tolist()
        if self._id_to_label:
            preds = [self._id_to_label[idx] for idx in preds]

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._preds['preds'], self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # accuracy
        preds = utils.transform(output_arrays[0], n_inputs)
        labels = self.data['label_ids']
        accuracy = np.mean(preds == labels)

        # loss
        losses = utils.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['accuracy'] = accuracy
        outputs['loss'] = loss

        return outputs



class BERTBinaryClassifier(BERTClassifier, ClassifierModule):
    ''' Multi-label classifier on BERT. '''
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 label_weight=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.label_weight = label_weight
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
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

    def _convert_y(self, y):
        try:
            label_set = set()
            for sample in y:
                _label_set = set()
                for _y in sample:
                    assert _y not in _label_set
                    label_set.add(_y)
                    _label_set.add(_y)
        except:
            raise ValueError(
                'The element of `y` should be a list of multiple answers. '
                'E.g. y=[[1, 3], [0], [0, 2]].')

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, (
                'Number of unique labels exceeds `label_size`.')
        else:
            self.label_size = len(label_set)

        # automatically set `id_to_label`
        if not self._id_to_label:
            self._id_to_label = list(label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._id_to_label = list(sorted(self._id_to_label))
            except Exception:
                pass
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {
            label: index for index, label in enumerate(self._id_to_label)}

        label_ids = [[1 if self._id_to_label[i] in sample else 0
                      for i in range(self.label_size)] for sample in y]
        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None, self.label_size], tf.int32),
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
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = BinaryCLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            label_weight=self.label_weight,
            scope='cls/seq_relationship',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_predict_ops(self):
        return [self._probs['probs']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))
        # probs
        probs = utils.transform(output_arrays[0], n_inputs)

        # preds
        preds = (probs >= 0.5)
        if self._id_to_label:
            preds = [[self._id_to_label[i] for i in range(self.label_size)
                      if _preds[i]] for _preds in preds]
        else:
            preds = [[i for i in range(self.label_size) if _preds[i]]
                     for _preds in preds]

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs

        return outputs



class BERTSeqClassifier(BERTClassifier, ClassifierModule):
    ''' Sequence labeling classifier on BERT. '''
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'
        if is_parallel:
            assert self.label_size, ('Can\'t parse data on multi-processing '
                'when `label_size` is None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        input_ids = []
        input_mask = []
        segment_ids = []

        # tokenize input texts
        for ex_id, example in enumerate(X_target):
            _input_tokens = self._convert_x(example, tokenized)

            utils.truncate_segments(
                [_input_tokens], self.max_seq_length,
                truncate_method=self.truncate_method)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _input_mask = [1 for _ in range(len(_input_tokens))]
            _segment_ids = [0 for _ in range(len(_input_tokens))]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _convert_x(self, x, tokenized):
        if not tokenized:
            raise ValueError(
                'Inputs of sequence classifier must be already tokenized '
                'and fed into `X_tokenized`.')

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return x

        # deal with tokenized and multiple inputs
        raise ValueError(
            'Sequence classifier does not support multi-segment inputs.')

    def _convert_y(self, y):
        try:
            label_set = set()
            for sample in y:
                for _y in sample:
                    label_set.add(_y)
        except:
            raise ValueError(
                'The element of `y` should be a list of labels.')

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, (
                'Number of unique `y`s exceeds `label_size`.')
        else:
            self.label_size = len(label_set)

        # automatically set `id_to_label`
        if not self._id_to_label:
            self._id_to_label = list(label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._id_to_label = list(sorted(self._id_to_label))
            except Exception:
                pass
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {
            label: index for index, label in enumerate(self._id_to_label)}

        label_ids = []
        for sample in y:
            sample = [label for label in sample]

            num_labels = len(sample)
            if num_labels < self.max_seq_length:
                sample.extend([0] * (self.max_seq_length - num_labels))
            elif num_labels > self.max_seq_length:
                sample = sample[:self.max_seq_length]

                utils.truncate_segments(
                    [sample], self.max_seq_length,
                    truncate_method=self.truncate_method)

            _label_ids = [self._label_to_id[label] for label in sample]
            label_ids.append(_label_ids)
        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None, self.max_seq_length], tf.int32),
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
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = SeqCLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/sequence',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['preds'], self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['input_mask'],
                        self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders['input_mask']]
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # accuracy
        batch_preds = output_arrays[1]
        accuracy = (np.sum((batch_preds == batch_labels) * batch_mask) /
                    batch_mask.sum())

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', accuracy %.4f' % accuracy
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['probs']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = utils.transform(output_arrays[0], n_inputs)

        # preds
        all_preds = np.argmax(probs, axis=-1)
        mask = self.data['input_mask']
        preds = []
        for _preds, _mask in zip(all_preds, mask):
            input_length = np.sum(_mask)
            _preds = _preds[:input_length].tolist()
            if self._id_to_label:
                _preds = [self._id_to_label[idx] for idx in _preds]
            preds.append(_preds)

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._preds['preds'], self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # accuracy
        preds = utils.transform(output_arrays[0], n_inputs)
        labels = self.data['label_ids']
        mask = self.data['input_mask']
        accuracy = (np.sum((preds == labels) * mask) /
                    mask.sum())

        # loss
        losses = utils.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['accuracy'] = accuracy
        outputs['loss'] = loss

        return outputs



class BERTNER(BERTClassifier, NERModule):
    ''' Named entity recognition on BERT. '''
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
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(NERModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._on_predict = False
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

    def predict(self, X=None, X_tokenized=None, batch_size=8):

        self._on_predict = True
        ret = super(NERModule, self).predict(
            X, X_tokenized, batch_size)
        self._on_predict = False

        return ret

    predict.__doc__ = NERModule.predict.__doc__

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
            is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            input_tokens, input_ids, input_mask, segment_ids = self._convert_X(
                X_target, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[utils.BACKUP_DATA + 'input_tokens'] = input_tokens
            data[utils.BACKUP_DATA + 'tokenized'] = [tokenized]
            data[utils.BACKUP_DATA + 'X_target'] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                raise ValueError(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ['[SEP]'])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_tokens, input_ids, input_mask, segment_ids

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        for ex_id, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if isinstance(_y, str):
                _entity_tokens = self.tokenizer.tokenize(_y)
                _entity_ids = [self.tokenizer.convert_tokens_to_ids(
                    _entity_tokens)]
            elif isinstance(_y, list):
                if isinstance(_y[0], str):
                    if tokenized:
                        _entity_ids = \
                            [self.tokenizer.convert_tokens_to_ids(_y)]
                    else:
                        _entity_ids = []
                        for _entity in _y:
                            _entity_ids.append(
                                self.tokenizer.convert_tokens_to_ids(
                                    self.tokenizer.tokenize(_entity)))
                elif isinstance(_y[0], list):
                    _entity_ids = []
                    for _entity in _y:
                        _entity_ids.append(
                            self.tokenizer.convert_tokens_to_ids(_entity))
            else:
                raise ValueError(
                    '`y` should be a list of entity strings.')

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]
            for _entity in _entity_ids:
                start_positions = utils.find_all_boyer_moore(
                    _input_ids, _entity)
                if not start_positions:
                    tf.logging.warning(
                        'Failed to find the mapping of entity to '
                        'inputs at line %d. A possible reason is '
                        'that the entity span is truncated due '
                        'to the `max_seq_length` setting.'
                        % (ex_id))
                    continue

                for start_position in start_positions:
                    end_position = start_position + len(_entity) - 1
                    if start_position == end_position:
                        _label_ids[start_position] = self.S_ID
                    else:
                        for i in range(start_position, end_position + 1):
                            _label_ids[i] = self.I_ID
                        _label_ids[start_position] = self.B_ID
                        _label_ids[end_position] = self.E_ID

            label_ids.append(_label_ids)

        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None, self.max_seq_length], tf.int32),
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
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = SeqCLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            label_size=5,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/sequence',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['preds'], self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['input_mask'],
                        self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders['input_mask']]
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # f1
        batch_preds = output_arrays[1]
        f1_token, f1_entity = self._get_f1(
            batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', f1/token %.4f' % f1_token
        info += ', f1/entity %.4f' % f1_entity
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['probs']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = utils.transform(output_arrays[0], n_inputs)

        # preds
        all_preds = np.argmax(probs, axis=-1)
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        mask = self.data['input_mask']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
        preds = []
        for i in range(len(all_preds)):
            _preds = all_preds[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _entities = self._get_entities(_preds[:_input_length])
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = ' '.join(_text)
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._preds['preds'], self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        preds = utils.transform(output_arrays[0], n_inputs)
        labels = self.data['label_ids']
        mask = self.data['input_mask']
        f1_token, f1_entity = self._get_f1(
            preds, labels, mask)

        # loss
        losses = utils.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['f1/token'] = f1_token
        outputs['f1/entity'] = f1_entity
        outputs['loss'] = loss

        return outputs



class BERTCRFNER(BERTNER, NERModule):
    ''' Named entity recognization on BERT with CRF. '''
    _INFER_ATTRIBUTES = BERTNER._INFER_ATTRIBUTES

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            label_size=5,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/sequence',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._probs['logits'], self._probs['transition_matrix'],
               self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['input_mask'],
                        self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders['input_mask']]
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # f1
        batch_logits = output_arrays[1]
        batch_transition_matrix = output_arrays[2]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(
                logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(
            batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[3]
        loss = np.mean(batch_losses)

        info = ''
        info += ', f1/token %.4f' % f1_token
        info += ', f1/entity %.4f' % f1_entity
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['logits'], self._probs['transition_matrix']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        logits = utils.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        mask = self.data['input_mask']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(
                _logits[:_input_length], transition_matrix)
            _entities = self._get_entities(_viterbi_seq)
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = ' '.join(_text)
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs['preds'] = preds
        outputs['logits'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._probs['logits'], self._probs['transition_matrix'],
                self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        logits = utils.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data['input_mask']
        labels = self.data['label_ids']
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(
                logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(
            preds, labels, mask)

        # loss
        losses = utils.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['f1/token'] = f1_token
        outputs['f1/entity'] = f1_entity
        outputs['loss'] = loss

        return outputs



class BERTCRFCascadeNER(BERTCRFNER, NERModule):
    ''' Named entity recognization and classification on BERT with CRF. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'entity_types': (
            'A list of strings that defines possible types of entities.'),
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
                 do_lower_case=True,
                 entity_types=None,
                 truncate_method='LIFO'):
        super(NERModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.entity_types = entity_types
        self._do_lower_case = do_lower_case
        self._on_predict = False
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
            is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'
        if is_parallel:
            assert self.entity_types, ('Can\'t parse data on multi-processing '
                'when `entity_types` is None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            input_tokens, input_ids, input_mask, segment_ids = self._convert_X(
                X_target, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[utils.BACKUP_DATA + 'input_tokens'] = input_tokens
            data[utils.BACKUP_DATA + 'tokenized'] = [tokenized]
            data[utils.BACKUP_DATA + 'X_target'] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        if not self.entity_types:
            type_B_id = {}
        else:
            type_B_id = {entity_type: 1 + 4 * i
                         for i, entity_type
                         in enumerate(self.entity_types)}
        for ex_id, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if not isinstance(_y, dict):
                raise ValueError(
                    'Wrong input format of `y`. An untokenized example: '
                    '`y = [{\'Person\': [\'Trump\', \'Obama\'], '
                    '\'City\': [\'Washington D.C.\'], ...}, ...]`')

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]

            # each type
            for _key in _y:

                # new type
                if _key not in type_B_id:
                    assert not self.entity_types, (
                        'Entity type `%s` not found in entity_types: %s.'
                        % (_key, self.entity_types))
                    type_B_id[_key] = 1 + 4 * len(list(type_B_id.keys()))
                _entities = _y[_key]

                # each entity
                for _entity in _entities:
                    if isinstance(_entity, str):
                        _entity_tokens = self.tokenizer.tokenize(_entity)
                        _entity_ids = \
                            self.tokenizer.convert_tokens_to_ids(
                                _entity_tokens)
                    elif isinstance(_entity, list):
                        if isinstance(_entity[0], str):
                            _entity_ids = \
                                self.tokenizer.convert_tokens_to_ids(_y)
                        else:
                            raise ValueError(
                                'Wrong input format (line %d): \'%s\'. '
                                % (ex_id, _entity))

                    # search and tag
                    start_positions = utils.find_all_boyer_moore(
                        _input_ids, _entity_ids)
                    if not start_positions:
                        tf.logging.warning(
                            'Failed to find the mapping of entity '
                            'to inputs at line %d. A possible '
                            'reason is that the entity span is '
                            'truncated due to the '
                            '`max_seq_length` setting.'
                            % (ex_id))
                        continue

                    for start_position in start_positions:
                        end_position = start_position + len(_entity) - 1
                        if start_position == end_position:
                            _label_ids[start_position] = type_B_id[_key] + 3
                        else:
                            for i in range(start_position, end_position + 1):
                                _label_ids[i] = type_B_id[_key] + 1
                            _label_ids[start_position] = type_B_id[_key]
                            _label_ids[end_position] = type_B_id[_key] + 2

            label_ids.append(_label_ids)
        if not self.entity_types:
            self.entity_types = list(type_B_id.keys())
        return label_ids

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            label_size=1 + len(self.entity_types) * 4,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/sequence',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._probs['logits'], self._probs['transition_matrix'],
               self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['input_mask'],
                        self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders['input_mask']]
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # f1
        batch_logits = output_arrays[1]
        batch_transition_matrix = output_arrays[2]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(
                logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(
            batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[3]
        loss = np.mean(batch_losses)

        info = ''
        for key in metrics:
            info += ', %s %.4f' % (key, metrics[key])
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['logits'], self._probs['transition_matrix']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        logits = utils.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        mask = self.data['input_mask']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(
                _logits[:_input_length], transition_matrix)

            if not tokenized:
                if isinstance(_text, list):
                    _text = ' '.join(_text)
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _tokens, _text, self._do_lower_case)

            _preds = {}
            for i, entity_type in enumerate(self.entity_types):
                _B_id = 1 + 4 * i
                _entities = self._get_entities(_viterbi_seq, _B_id)
                if _entities:
                    _preds[entity_type] = []

                    for _entity in _entities:
                        _start, _end = _entity[0], _entity[1]
                        if tokenized:
                            _entity_tokens = _tokens[_start: _end + 1]
                            _preds[entity_type].append(_entity_tokens)
                        else:
                            try:
                                _text_start = _mapping_start[_start]
                                _text_end = _mapping_end[_end]
                            except:
                                continue
                            _entity_text = _text[_text_start: _text_end]
                            _preds[entity_type].append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs['preds'] = preds
        outputs['logits'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._probs['logits'], self._probs['transition_matrix'],
                self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        logits = utils.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data['input_mask']
        labels = self.data['label_ids']
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(
                logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(
            preds, labels, mask)

        # loss
        losses = utils.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        for key in metrics:
            outputs[key] = metrics[key]
        outputs['loss'] = loss

        return outputs



class BERTMRC(BERTClassifier, MRCModule):
    ''' Machine reading comprehension on BERT. '''
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
                 max_seq_length=256,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='longer-FO'):
        super(MRCModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._on_predict = False
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

    def predict(self, X=None, X_tokenized=None,
                batch_size=8):

        self._on_predict = True
        ret = super(MRCModule, self).predict(
            X, X_tokenized, batch_size)
        self._on_predict = False

        return ret

    predict.__doc__ = MRCModule.predict.__doc__

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, segment_ids,
             doc_ids, doc_text, doc_start) = self._convert_X(
                 X_target, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[utils.BACKUP_DATA + 'input_tokens'] = input_tokens
            data[utils.BACKUP_DATA + 'tokenized'] = [tokenized]
            data[utils.BACKUP_DATA + 'X_target'] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(
                y, doc_ids, doc_text, doc_start, tokenized)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                raise ValueError(
                    'Wrong input format (line %d): \'%s\'. '
                    'An untokenized example: '
                    '`X = [{\'doc\': \'...\', \'question\': \'...\', ...}, '
                    '...]`' % (ex_id, example))

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            _doc_tokens = segments.pop('doc')
            segments = list(segments.values()) + [_doc_tokens]
            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)
            _doc_tokens = segments[-1]

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ['[SEP]'])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))
            _doc_start = len(_input_tokens) - len(_doc_tokens) - 1

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _doc_ids = _input_ids[_doc_start: -1]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[ex_id]['doc'])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, segment_ids,
                doc_ids, doc_text, doc_start)

    def _convert_x(self, x, tokenized):
        output = {}

        assert isinstance(x, dict) and 'doc' in x

        for key in x:
            if not tokenized:
                # deal with general inputs
                output[key] = self.tokenizer.tokenize(x[key])
                continue

            # deal with tokenized inputs
            output[key] = x[key]

        return output

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []

        invalid_ids = []
        for ex_id, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                continue

            if not isinstance(_y, dict) or \
                    'text' not in _y or 'answer_start' not in _y:
                raise ValueError(
                    'Wrong input format of `y`. An untokenized example: '
                    '`y = [{\'text\': \'Obama\', \'answer_start\': 12}, '
                    'None, ...]`')

            _answer_text = _y['text']
            _answer_start = _y['answer_start']
            _doc_ids = doc_ids[ex_id]

            # tokenization
            if isinstance(_answer_text, str):
                _answer_tokens = self.tokenizer.tokenize(_answer_text)
                _answer_ids = self.tokenizer.convert_tokens_to_ids(
                    _answer_tokens)
            elif isinstance(_answer_text, list):
                assert tokenized, (
                    '%s does not support multiple answer spans.'
                    % self.__class__.__name__)
                _answer_tokens = _answer_text
                _answer_ids = self.tokenizer.convert_tokens_to_ids(
                    _answer_text)
            else:
                raise ValueError(
                    'Invalid answer text at line %d: `%s`.'
                    % (ex_id, _answer_text))

            if isinstance(_answer_text, str):
                _overlap_time = len(utils.find_all_boyer_moore(
                    doc_text[ex_id][:_answer_start], _answer_text))
                try:
                    start_position = utils.find_all_boyer_moore(
                        _doc_ids, _answer_ids)[_overlap_time]
                    end_position = start_position + len(_answer_ids) - 1
                except IndexError:
                    label_ids.append([0, 0])
                    invalid_ids.append(ex_id)
                    continue
            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning(
                        'Wrong `answer_start` at line %d. Ignored and set '
                        'label to null text.')
                    label_ids.append([0, 0])
                    invalid_ids.append(ex_id)
                    continue

            label_ids.append([start_position + doc_start[ex_id],
                              end_position + doc_start[ex_id]])

        if invalid_ids:
            tf.logging.warning(
                'Failed to find the mapping of answer to inputs at '
                'line %s. A possible reason is that the answer spans '
                'are truncated due to the `max_seq_length` setting.'
                % invalid_ids)

        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None, 2], tf.int32),
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
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='mrc',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['preds'], self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # exact match & f1
        batch_preds = output_arrays[1]
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', exact_match %.4f' % exact_match
        info += ', f1 %.4f' % f1
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['probs'], self._preds['preds']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = utils.transform(output_arrays[0], n_inputs)

        # preds
        batch_preds = utils.transform(output_arrays[1], n_inputs)
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
        preds = []
        for ex_id, _preds in enumerate(batch_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if _start == 0 or _end == 0 or _start > _end:
                preds.append(None)
                continue
            _tokens = tokens[ex_id]

            if tokenized:
                _span_tokens = _tokens[_start: _end + 1]
                preds.append(_span_tokens)
            else:
                _sample = text[ex_id]
                _text = [_sample[key] for key in _sample if key != 'doc']
                _text.append(_sample['doc'])
                _text = ' '.join(_text)
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _tokens, _text, self._do_lower_case)

                try:
                    _text_start = _mapping_start[_start]
                    _text_end = _mapping_end[_end]
                except:
                    preds.append(None)
                    continue
                _span_text = _text[_text_start: _text_end]
                preds.append(_span_text)

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs

        return outputs

    def _get_score_ops(self):
        return [self._preds['preds'], self._losses['losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # exact match & f1
        batch_preds = utils.transform(output_arrays[0], n_inputs)
        batch_labels = self.data['label_ids']
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        losses = utils.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['exact_match'] = exact_match
        outputs['f1'] = f1
        outputs['loss'] = loss

        return outputs



class BERTVerifierMRC(BERTMRC, MRCModule):
    ''' Machine reading comprehension on BERT, with a external front
    verifier. '''
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=256,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 drop_pooler=False,
                 truncate_method='longer-FO'):
        super(MRCModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._drop_pooler = drop_pooler
        self._on_predict = False
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
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, segment_ids,
             doc_ids, doc_text, doc_start) = self._convert_X(
                 X_target, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[utils.BACKUP_DATA + 'input_tokens'] = input_tokens
            data[utils.BACKUP_DATA + 'tokenized'] = [tokenized]
            data[utils.BACKUP_DATA + 'X_target'] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids, has_answer = self._convert_y(
                y, doc_ids, doc_text, doc_start, tokenized)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)
            data['has_answer'] = np.array(has_answer, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []
        has_answer = []

        invalid_ids = []
        for ex_id, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                has_answer.append(0)
                continue

            if not isinstance(_y, dict) or \
                    'text' not in _y or 'answer_start' not in _y:
                raise ValueError(
                    'Wrong input format of `y`. An untokenized example: '
                    '`y = [{\'text\': \'Obama\', \'answer_start\': 12}, '
                    'None, ...]`')

            _answer_text = _y['text']
            _answer_start = _y['answer_start']
            _doc_ids = doc_ids[ex_id]

            # tokenization
            if isinstance(_answer_text, str):
                _answer_tokens = self.tokenizer.tokenize(_answer_text)
                _answer_ids = self.tokenizer.convert_tokens_to_ids(
                    _answer_tokens)
            elif isinstance(_answer_text, list):
                assert tokenized, (
                    '%s does not support multiple answer spans.'
                    % self.__class__.__name__)
                _answer_tokens = _answer_text
                _answer_ids = self.tokenizer.convert_tokens_to_ids(
                    _answer_text)
            else:
                raise ValueError(
                    'Invalid answer text at line %d: `%s`.'
                    % (ex_id, _answer_text))

            if isinstance(_answer_text, str):
                _overlap_time = len(utils.find_all_boyer_moore(
                    doc_text[ex_id][:_answer_start], _answer_text))

                start_positions = utils.find_all_boyer_moore(
                    _doc_ids, _answer_ids)
                if _overlap_time >= len(start_positions):
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(ex_id)
                    continue
                start_position = start_positions[_overlap_time]
                end_position = start_position + len(_answer_ids) - 1

            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning(
                        'Wrong `answer_start` at line %d. Ignored and set '
                        'label to null text.')
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(ex_id)
                    continue

            label_ids.append([start_position + doc_start[ex_id],
                              end_position + doc_start[ex_id]])
            has_answer.append(1)

        if invalid_ids:
            tf.logging.warning(
                'Failed to find the mapping of answer to inputs at '
                'line %s. A possible reason is that the answer spans '
                'are truncated due to the `max_seq_length` setting.'
                % invalid_ids)

        return label_ids, has_answer

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids',
                [None, 2], tf.int32),
            'has_answer': utils.get_placeholder(
                target, 'has_answer',
                [None], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = utils.get_placeholder(
                target, 'sample_weight',
                [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
            **kwargs)
        verifier = CLSDecoder(
            is_training=is_training,
            input_tensor=encoder.get_pooled_output(),
            label_ids=split_placeholders['has_answer'],
            label_size=2,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/verifier',
            **kwargs)
        if is_training:
            sample_weight = (
                tf.cast(split_placeholders['has_answer'], tf.float32) *
                split_placeholders.get('sample_weight'))
        else:
            sample_weight = split_placeholders.get('sample_weight')
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder.get_sequence_output(),
            label_ids=split_placeholders['label_ids'],
            sample_weight=sample_weight,
            scope='mrc',
            **kwargs)

        (verifier_total_loss, verifier_losses, verifier_probs,
         verifier_preds) = verifier.get_forward_outputs()
        (decoder_total_loss, decoder_losses, decoder_probs,
         decoder_preds) = decoder.get_forward_outputs()

        total_loss = verifier_total_loss + decoder_total_loss
        losses = collections.OrderedDict()
        probs = collections.OrderedDict()
        preds = collections.OrderedDict()
        for key in verifier_losses:
            losses['verifier_' + key] = verifier_losses[key]
        for key in verifier_probs:
            probs['verifier_' + key] = verifier_probs[key]
        for key in verifier_preds:
            preds['verifier_' + key] = verifier_preds[key]
        for key in decoder_losses:
            losses['mrc_' + key] = decoder_losses[key]
        for key in decoder_probs:
            probs['mrc_' + key] = decoder_probs[key]
        for key in decoder_preds:
            preds['mrc_' + key] = decoder_preds[key]

        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['verifier_preds'],
               self._losses['verifier_losses'],
               self._preds['mrc_preds'],
               self._losses['mrc_losses']]
        if as_feature:
            ops.extend([self.placeholders['label_ids']])
            ops.extend([self.placeholders['has_answer']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-2]
            batch_has_answer = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders['label_ids']]
            batch_has_answer = feed_dict[
                self.placeholders['has_answer']]

        # verifier accuracy
        batch_has_answer_preds = output_arrays[1]
        has_answer_accuracy = np.mean(
            batch_has_answer_preds == batch_has_answer)

        # verifier loss
        batch_verifier_losses = output_arrays[2]
        verifier_loss = np.mean(batch_verifier_losses)

        # mrc exact match & f1
        batch_preds = output_arrays[3]
        for i in range(len(batch_has_answer_preds)):
            if batch_has_answer_preds[i] == 0:
                batch_preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # mrc loss
        batch_losses = output_arrays[4]
        loss = np.mean(batch_losses)

        info = ''
        info += ', has_ans_accuracy %.4f' % has_answer_accuracy
        info += ', exact_match %.4f' % exact_match
        info += ', f1 %.4f' % f1
        info += ', verifier_loss %.6f' % verifier_loss
        info += ', mrc_loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['verifier_probs'],
                self._preds['verifier_preds'],
                self._probs['mrc_probs'],
                self._preds['mrc_preds']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier preds & probs
        verifier_probs = utils.transform(output_arrays[0], n_inputs)[:, 1]
        verifier_preds = utils.transform(output_arrays[1], n_inputs)

        # mrc preds & probs
        probs = utils.transform(output_arrays[2], n_inputs)
        mrc_preds = utils.transform(output_arrays[3], n_inputs)
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
        preds = []
        for ex_id, _preds in enumerate(mrc_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if verifier_preds[ex_id] == 0 or _start == 0 or _end == 0 \
                    or _start > _end:
                preds.append(None)
                continue
            _tokens = tokens[ex_id]

            if tokenized:
                _span_tokens = _tokens[_start: _end + 1]
                preds.append(_span_tokens)
            else:
                _sample = text[ex_id]
                _text = [_sample[key] for key in _sample if key != 'doc']
                _text.append(_sample['doc'])
                _text = ' '.join(_text)
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _tokens, _text, self._do_lower_case)

                try:
                    _text_start = _mapping_start[_start]
                    _text_end = _mapping_end[_end]
                except:
                    preds.append(None)
                    continue
                _span_text = _text[_text_start: _text_end]
                preds.append(_span_text)

        outputs = {}
        outputs['verifier_probs'] = verifier_probs
        outputs['verifier_preds'] = verifier_preds
        outputs['mrc_probs'] = probs
        outputs['mrc_preds'] = preds

        return outputs

    def _get_score_ops(self):
        return [self._preds['verifier_preds'],
                self._losses['verifier_losses'],
                self._preds['mrc_preds'],
                self._losses['mrc_losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier accuracy
        has_answer_preds = utils.transform(output_arrays[0], n_inputs)
        has_answer_accuracy = np.mean(
            has_answer_preds == self.data['has_answer'])

        # verifier loss
        verifier_losses = utils.transform(output_arrays[1], n_inputs)
        verifier_loss = np.mean(verifier_losses)

        # mrc exact match & f1
        preds = utils.transform(output_arrays[2], n_inputs)
        for i in range(len(has_answer_preds)):
            if has_answer_preds[i] == 0:
                preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(preds, self.data['label_ids'])

        # mrc loss
        losses = utils.transform(output_arrays[3], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['has_ans_accuracy'] = has_answer_accuracy
        outputs['exact_match'] = exact_match
        outputs['f1'] = f1
        outputs['verifier_loss'] = verifier_loss
        outputs['mrc_loss'] = loss

        return outputs



class BERTLM(LMModule):
    ''' Language modeling on BERT. '''
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
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(BERTLM, self).__init__(
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
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            if y is not None:
                assert not self.do_sample_next_sentence, (
                    '`y` should be None when `do_sample_next_sentence` is True.')
            else:
                assert self.do_sample_next_sentence, (
                    '`y` can\'t be None when `do_sample_next_sentence` is False.')

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

            if is_training and self.do_sample_next_sentence:
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
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
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
        if is_training and self.do_sample_next_sentence:
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

    def _convert_x(self, x, tokenized):
        if not tokenized:
            # deal with general inputs
            if isinstance(x, str):
                return [self.tokenizer.tokenize(x)]

            # deal with multiple inputs
            return [self.tokenizer.tokenize(seg) for seg in x]

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return [x]

        # deal with tokenized and multiple inputs
        return x

    def _convert_y(self, y):
        label_set = set(y)

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, (
                'Number of unique `y`s exceeds 2.')
        else:
            self.label_size = len(label_set)

        # automatically set `id_to_label`
        if not self._id_to_label:
            self._id_to_label = list(label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._id_to_label = list(sorted(self._id_to_label))
            except Exception:
                pass
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {
            label: index for index, label in enumerate(self._id_to_label)}

        label_ids = [self._label_to_id[label] for label in y]
        return label_ids

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
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
        all_preds = utils.transform(output_arrays[0], n_inputs).tolist()
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
                                   short_seq_prob, vocab_words):
    document = all_documents[document_index]

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_seq_length
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_seq_length)

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
        if i == len(document) - 1 or current_length >= target_seq_length:
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
                    target_b_length = target_seq_length - len(tokens_a)

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


def get_bert_config(config_file=None):
    if not os.path.exists(config_file):
        raise ValueError(
            'Can\'t find config_file \'%s\'. '
            'Please pass the correct path of configuration file, '
            'e.g.`bert_config.json`. An example can be downloaded from '
            'https://github.com/google-research/bert.' % config_file)
    return BERTConfig.from_json_file(config_file)


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 2,
        '/pooler/': 1,
        'cls/': 0,
        'mrc/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/layer_%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
