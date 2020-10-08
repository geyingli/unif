# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Applications based on RoBERTa. '''

import copy
import numpy as np

from uf.tools import tf
from .base import ClassifierModule, MRCModule, LMModule
from uf.modeling.bert import BERTEncoder, BERTDecoder
from .bert import (BERTClassifier, BERTBinaryClassifier, BERTSeqClassifier,
                   BERTMRC, BERTLM,
                   get_bert_config, get_word_piece_tokenizer,
                   get_key_to_depths, create_masked_lm_predictions)
import uf.utils as utils


class RoBERTaClassifier(BERTClassifier, ClassifierModule):
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES


class RoBERTaBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    _INFER_ATTRIBUTES = BERTBinaryClassifier._INFER_ATTRIBUTES


class RoBERTaSeqClassifier(BERTSeqClassifier, ClassifierModule):
    _INFER_ATTRIBUTES = BERTSeqClassifier._INFER_ATTRIBUTES


class RoBERTaMRC(BERTMRC, MRCModule):
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES


class RoBERTaLM(BERTLM, LMModule):
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
                 dupe_factor=1,
                 masked_lm_prob=0.15,
                 do_whole_word_mask=True,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = 2
        self._drop_pooler = drop_pooler
        self._max_predictions_per_seq = max_predictions_per_seq
        self._dupe_factor = dupe_factor
        self._masked_lm_prob = masked_lm_prob
        self._do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, (
            'RoBERTa is a masked language modeling model. '
            '`y` should be None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights) = \
                 self._convert_X(X_tokenized if tokenized else X,
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

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight (fit)
        if is_training:
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

        # If `max_seq_length` is not mannually assigned,
        # the value will be set to the maximum length of
        # `input_ids`.
        if not self.max_seq_length:
            max_seq_length = 0
            for segments in segment_input_tokens:
                # subtract `[CLS]` and `[SEP]s`
                seq_length = sum([len(seg) + 1 for seg in segments]) + 1
                max_seq_length = max(max_seq_length, seq_length)
            self.max_seq_length = max_seq_length
            tf.logging.info('Adaptive max_seq_length: %d'
                            % self.max_seq_length)

        input_ids = []
        input_mask = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []

        # duplicate raw inputs
        if is_training and self._dupe_factor > 1:
            new_segment_input_tokens = []
            for _ in range(self._dupe_factor):
                new_segment_input_tokens.extend(
                    copy.deepcopy(segment_input_tokens))
            segment_input_tokens = new_segment_input_tokens

        # random sampling of next sentence
        if is_training:
            new_segment_input_tokens = []
            for ex_id in range(len(segment_input_tokens)):
                instances = create_instances_from_document(
                    all_documents=segment_input_tokens,
                    document_index=ex_id,
                    max_seq_length=self.max_seq_length - 2,
                    masked_lm_prob=self._masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    vocab_words=list(self.tokenizer.vocab.keys()))
                for segments in instances:
                    new_segment_input_tokens.append(segments)
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
                (_input_tokens, _masked_lm_positions, _masked_lm_labels) = \
                    create_masked_lm_predictions(
                        tokens=_input_tokens,
                        masked_lm_prob=self._masked_lm_prob,
                        max_predictions_per_seq=self._max_predictions_per_seq,
                        vocab_words=list(self.tokenizer.vocab.keys()),
                        do_whole_word_mask=self._do_whole_word_mask)
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
                masked_lm_positions, masked_lm_ids, masked_lm_weights)

    def _set_placeholders(self, target, on_export=False):
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
            sample_weight=split_placeholders.get('sample_weight'),
            scope_lm='cls/predictions',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['MLM'], self._losses['MLM']]
        if as_feature:
            ops.extend(
                [self.placeholders['masked_lm_positions'],
                 self.placeholders['masked_lm_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mlm_positions = output_arrays[-2]
            batch_mlm_labels = output_arrays[-1]
        else:
            batch_mlm_positions = \
                feed_dict[self.placeholders['masked_lm_positions']]
            batch_mlm_labels = \
                feed_dict[self.placeholders['masked_lm_ids']]

        # MLM accuracy
        batch_mlm_preds = output_arrays[1]
        batch_mlm_positions = np.reshape(batch_mlm_positions, [-1])
        batch_mlm_labels = np.reshape(batch_mlm_labels, [-1])
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

        return info

    def _get_predict_ops(self):
        return [self._preds['MLM']]

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

        outputs = {}
        outputs['mlm_preds'] = mlm_preds

        return outputs


def create_instances_from_document(all_documents, document_index,
                                   max_seq_length, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words):
    document = all_documents[document_index]
    instances = []

    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.extend(segment)
        current_length += len(segment)
        i += 1
        if current_length >= max_seq_length:
            instances.append([current_chunk])
            current_chunk = []
            current_length = 0
    if current_chunk:
        instances.append([current_chunk])

    return instances