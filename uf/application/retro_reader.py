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
''' Applications based on Retro-Reader. '''

import numpy as np

from uf.tools import tf
from .base import MRCModule
from .bert import BERTVerifierMRC, get_bert_config
from .albert import get_albert_config
from uf.modeling.bert import BERTEncoder
from uf.modeling.albert import ALBERTEncoder
from uf.modeling.retro_reader import RetroReaderDecoder
from uf.tokenization.word_piece import get_word_piece_tokenizer
import uf.utils as utils



class RetroReaderMRC(BERTVerifierMRC, MRCModule):
    ''' Machine reading comprehension on Retro-Reader. '''
    _INFER_ATTRIBUTES = BERTVerifierMRC._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=256,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 reading_module='bert',
                 matching_mechanism='cross-attention',
                 beta_1=0.5,
                 beta_2=0.5,
                 threshold=1.0,
                 truncate_method='longer-FO'):
        super(MRCModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._do_lower_case = do_lower_case
        self._on_predict = False
        self._reading_module = reading_module
        self._matching_mechanism = matching_mechanism
        self._threshold = threshold
        self.__init_args__ = locals()

        if reading_module == 'albert':
            self.bert_config = get_albert_config(config_file)
        else:
            self.bert_config = get_bert_config(config_file)

        assert reading_module in ('bert', 'roberta', 'albert', 'electra'), (
            'Invalid value of `reading_module`: %s. Pick one from '
            '`bert`, `roberta`, `albert` and `electra`.')
        assert matching_mechanism in (
            'cross-attention', 'matching-attention'), (
                'Invalid value of `matching_machanism`: %s. Pick one from '
                '`cross-attention` and `matching-attention`.')
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
            (input_tokens, input_ids, input_mask, query_mask, segment_ids,
             doc_ids, doc_text, doc_start) = self._convert_X(
                X_target, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['query_mask'] = np.array(query_mask, dtype=np.int32)
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
        query_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _query_mask = [1]
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
                if s_id == 0:
                    _query_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))
            _doc_start = len(_input_tokens) - len(_doc_tokens) - 1
            
            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _doc_ids = _input_ids[_doc_start: -1]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)
            for _ in range(self.max_seq_length - len(_query_mask)):
                _query_mask.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            query_mask.append(_query_mask)
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[ex_id]['doc'])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, query_mask, segment_ids,
                doc_ids, doc_text, doc_start)

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'query_mask': utils.get_placeholder(
                target, 'query_mask',
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

        def _get_encoder(model_name):
            if model_name == 'bert' or model_name == 'roberta':
                sketchy_encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='bert',
                    **kwargs)
            elif model_name == 'albert':
                sketchy_encoder = ALBERTEncoder(
                    albert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='bert',
                    **kwargs)
            elif model_name == 'electra':
                sketchy_encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='electra',
                    **kwargs)
            return sketchy_encoder

        sketchy_encoder = _get_encoder(self._reading_module)
        intensive_encoder = sketchy_encoder    #TODO: experiment with different encoder
        decoder = RetroReaderDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            sketchy_encoder=sketchy_encoder,
            intensive_encoder=intensive_encoder,
            query_mask=split_placeholders['query_mask'],
            label_ids=split_placeholders['label_ids'],
            has_answer=split_placeholders['has_answer'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='retro_reader',
            matching_mechanism=self._matching_mechanism,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            threshold=self._threshold,
            trainable=True,
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['verifier_preds'],
               self._preds['mrc_preds'],
               self._losses['sketchy_losses'],
               self._losses['intensive_losses']]
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

        # mrc exact match & f1
        batch_preds = output_arrays[2]
        for i in range(len(batch_has_answer_preds)):
            if batch_has_answer_preds[i] == 0:
                batch_preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # sketchy loss
        batch_sketchy_losses = output_arrays[3]
        sketchy_loss = np.mean(batch_sketchy_losses)

        # intensive loss
        batch_intensive_losses = output_arrays[4]
        intensive_loss = np.mean(batch_intensive_losses)

        info = ''
        info += ', has_ans_accuracy %.4f' % has_answer_accuracy
        info += ', exact_match %.4f' % exact_match
        info += ', f1 %.4f' % f1
        info += ', sketchy_loss %.6f' % sketchy_loss
        info += ', intensive_loss %.6f' % intensive_loss

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
        verifier_probs = utils.transform(output_arrays[0], n_inputs)
        verifier_preds = utils.transform(output_arrays[1], n_inputs)

        # mrc preds & probs
        preds = []
        probs = utils.transform(output_arrays[2], n_inputs)
        mrc_preds = utils.transform(output_arrays[3], n_inputs)
        tokens = self.data[utils.BACKUP_DATA + 'input_tokens']
        text = self.data[utils.BACKUP_DATA + 'X_target']
        tokenized = self.data[utils.BACKUP_DATA + 'tokenized'][0]
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
                self._preds['mrc_preds'],
                self._losses['sketchy_losses'],
                self._losses['intensive_losses']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier accuracy
        has_answer_preds = utils.transform(output_arrays[0], n_inputs)
        has_answer_accuracy = np.mean(
            has_answer_preds == self.data['has_answer'])

        # mrc exact match & f1
        preds = utils.transform(output_arrays[1], n_inputs)
        for i in range(len(has_answer_preds)):
            if has_answer_preds[i] == 0:
                preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(preds, self.data['label_ids'])

        # sketchy loss
        sketchy_losses = utils.transform(output_arrays[2], n_inputs)
        sketchy_loss = np.mean(sketchy_losses)

        # intensive loss
        intensive_losses = utils.transform(output_arrays[3], n_inputs)
        intensive_loss = np.mean(intensive_losses)

        outputs = {}
        outputs['has_ans_accuracy'] = has_answer_accuracy
        outputs['exact_match'] = exact_match
        outputs['f1'] = f1
        outputs['sketchy_loss'] = sketchy_loss
        outputs['intensive_loss'] = intensive_loss

        return outputs


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 2,
        '/pooler/': 1,
        'retro_reader/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/layer_%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
