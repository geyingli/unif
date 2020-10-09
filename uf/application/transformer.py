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
import numpy as np

from uf.tools import tf
from .base import MTModule
from uf.modeling.transformer import Transformer
from uf.tokenization.word_piece import WordPieceTokenizer
import uf.utils as utils



class TransformerMT(MTModule):
    ''' Machine translation on Transformer. '''
    _INFER_ATTRIBUTES = {
        'source_max_seq_length': (
            'An integer that defines max sequence length of source language '
            'tokens, which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'target_max_seq_length': (
            'An integer that defines max sequence length of target language '
            'tokens, which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'init_checkpoint': (
            'A string that directs to the checkpoint file used for '
            'initialization')}

    def __init__(self,
                 vocab_file,
                 source_max_seq_length=64,
                 target_max_seq_length=64,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 hidden_size=768,
                 num_hidden_layers=6,
                 num_attention_heads=12,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(MTModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.source_max_seq_length = source_max_seq_length
        self.target_max_seq_length = target_max_seq_length
        self.truncate_method = truncate_method
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._id_to_label = None
        self.__init_args__ = locals()

        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, '`y` can\'t be None.'
        if '<s>' not in self.tokenizer.vocab:
            self.tokenizer.add('<s>')
        if '</s>' not in self.tokenizer.vocab:
            self.tokenizer.add('</s>')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            source_ids = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized)
            data['source_ids'] = np.array(source_ids, dtype=np.int32)
            n_inputs = len(source_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            target_ids = self._convert_y(y, n_inputs)
            data['target_ids'] = np.array(target_ids, dtype=np.int32)

            # convert sample_weight (fit, score)
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        source_ids = []

        for ex_id, example in enumerate(X_target):
            try:
                _source_tokens = self._convert_x(example, tokenized)
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))
            _source_ids = self.tokenizer.convert_tokens_to_ids(_source_tokens)

            utils.truncate_segments(
                [_source_ids], self.source_max_seq_length,
                truncate_method=self.truncate_method)

            if len(_source_ids) < self.source_max_seq_length:
                _source_ids.extend([0 for _ in range(
                    self.source_max_seq_length - len(_source_ids))])
            source_ids.append(_source_ids)

        return source_ids

    def _convert_x(self, x, tokenized):
        if not tokenized:
            # deal with general inputs
            if isinstance(x, str):
                return self.tokenizer.tokenize(x)

        # deal with tokenized inputs
        elif isinstance(x[0], str):
            return x

        # deal with tokenized and multiple inputs
        raise ValueError(
            'Machine translation module only supports single sentence inputs.')

    def _convert_y(self, y, n_inputs=None):
        target_ids = []
        sos_id = self.tokenizer.convert_tokens_to_ids(['<s>'])[0]
        eos_id = self.tokenizer.convert_tokens_to_ids(['</s>'])[0]

        for _y in y:
            if isinstance(_y, str):
                _target_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(_y))
            elif isinstance(_y, list):
                assert isinstance(_y[0], str), (
                    'Machine translation module only supports '
                    'single sentence inputs.')
                _target_ids = self.tokenizer.convert_tokens_to_ids(_y)

            utils.truncate_segments(
                [_target_ids], self.target_max_seq_length - 2,
                truncate_method=self.truncate_method)
            _target_ids = [sos_id] + _target_ids + [eos_id]

            if len(_target_ids) < self.target_max_seq_length:
                _target_ids.extend([0 for _ in range(
                    self.target_max_seq_length - len(_target_ids))])
            target_ids.append(_target_ids)

        return target_ids

    def _set_placeholders(self, target, on_export=False):
        self.placeholders = {
            'source_ids': utils.get_placeholder(
                target, 'source_ids',
                [None, self.source_max_seq_length], tf.int32),
            'target_ids': utils.get_placeholder(
                target, 'target_ids',
                [None, self.target_max_seq_length], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = Transformer(
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            source_ids=split_placeholders['source_ids'],
            target_ids=split_placeholders['target_ids'],
            sos_id=self.tokenizer.convert_tokens_to_ids(['<s>'])[0],
            sample_weight=split_placeholders.get('sample_weight'),
            hidden_size=self._hidden_size,
            num_blocks=self._num_hidden_layers,
            num_attention_heads=self._num_attention_heads,
            scope='transformer',
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['MT'], self._losses['MT']]
        if as_feature:
            ops.extend([self.placeholders['target_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_target = output_arrays[-1]
        else:
            batch_target = feed_dict[self.placeholders['target_ids']]

        # accuracy
        batch_preds = output_arrays[1]
        batch_labels = np.hstack(
            (batch_target[:, 1:], np.zeros((self.batch_size, 1))))
        batch_mask = (batch_labels > 0)
        accuracy = np.sum((batch_preds == batch_labels) * batch_mask) / \
            np.sum(batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', accuracy %.4f' % accuracy
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._preds['MT']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        all_preds = utils.transform(output_arrays[0], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            for i in range(self.target_max_seq_length):
                if _pred_tokens[i] == '</s>':
                    _pred_tokens = _pred_tokens[:i]
                    break
            _pred_text = utils.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs['preds'] = preds

        return outputs

    def _get_score_ops(self):
        return [self._preds['MT'], self._losses['MT']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # accuracy
        preds = utils.transform(output_arrays[0], n_inputs)
        target = self.data['target_ids']
        labels = np.hstack(
            (target[:, 1:], np.zeros((n_inputs, 1))))
        mask = (labels > 0)
        accuracy = np.sum((preds == labels) * mask) / \
            np.sum(mask)

        # bleu
        bleu = self._get_bleu(preds, labels, mask)

        # rouge
        rouge = self._get_rouge(preds, labels, mask)

        # loss
        losses = utils.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs['accuracy'] = accuracy
        outputs['bleu'] = bleu
        outputs['rouge'] = rouge
        outputs['loss'] = loss

        return outputs



def get_word_piece_tokenizer(vocab_file, do_lower_case=True):
    if not os.path.exists(vocab_file):
        raise ValueError(
            'Can\'t find vocab_file \'%s\'. '
            'Please pass the correct path of vocabulary file, '
            'e.g.`vocab.txt`.' % vocab_file)
    return WordPieceTokenizer(vocab_file, do_lower_case=do_lower_case)


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 1,
        'cls': 0,}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/block_%d/' % layer_idx] = num_hidden_layers - layer_idx
    return key_to_depths
