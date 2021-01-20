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
''' Applications based on RecBERT. '''

import random
import numpy as np

from uf.tools import tf
from .base import LMModule
from .bert import get_bert_config, get_word_piece_tokenizer, get_key_to_depths
from uf.modeling.rec_bert import RecBERT
import uf.utils as utils



class RecBERTLM(LMModule):
    ''' Language modeling on RecBERT. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + 1'),
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
                 replace_prob=0.05,
                 add_prob=0.05,
                 subtract_prob=0.05,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._replace_prob = replace_prob
        self._add_prob = add_prob
        self._subtract_prob = subtract_prob
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, ('%s is unsupervised. `y` should be None.'
                           % self.__class__.__name__)

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            (input_ids, replace_label_ids,
             add_label_ids, subtract_label_ids) = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized,
                is_training=is_training)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)

            if is_training:
                data['replace_label_ids'] = np.array(
                    replace_label_ids, dtype=np.int32)
                data['add_label_ids'] = np.array(
                    add_label_ids, dtype=np.int32)
                data['subtract_label_ids'] = np.array(
                    subtract_label_ids, dtype=np.int32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized, is_training):
        input_ids = []
        replace_label_ids = []
        add_label_ids = []
        subtract_label_ids = []

        for ex_id, example in enumerate(X_target):
            _input_tokens = self._convert_x(example, tokenized)

            utils.truncate_segments(
                [_input_tokens], self.max_seq_length,
                truncate_method=self.truncate_method)

            _input_ids = self.tokenizer.convert_tokens_to_ids(
                _input_tokens)
            nonpad_seq_length = len(_input_ids)
            for _ in range(self.max_seq_length - nonpad_seq_length):
                _input_ids.append(0)

            _replace_label_ids = []
            _add_label_ids = []
            _subtract_label_ids = []

            # replace/add/subtract
            if is_training:
                for _input_id in _input_ids:
                    _replace_label_ids.append(_input_id)
                    _add_label_ids.append(0)
                    _subtract_label_ids.append(0)

                max_replace = int(nonpad_seq_length * self._replace_prob)
                max_add = int(nonpad_seq_length * self._add_prob)
                max_subtract = int(nonpad_seq_length * self._subtract_prob)

                sample_wrong_tokens(
                    _input_ids, _replace_label_ids,
                    _add_label_ids, _subtract_label_ids,
                    max_replace, max_add, max_subtract,
                    nonpad_seq_length=nonpad_seq_length,
                    vocab_size=len(self.tokenizer.vocab))

            input_ids.append(_input_ids)
            replace_label_ids.append(_replace_label_ids)
            add_label_ids.append(_add_label_ids)
            subtract_label_ids.append(_subtract_label_ids)

        return input_ids, replace_label_ids, add_label_ids, subtract_label_ids

    def _convert_x(self, x, tokenized):
        try:
            if not tokenized:
                # deal with general inputs
                if isinstance(x, str):
                    return self.tokenizer.tokenize(x)

            # deal with tokenized inputs
            elif isinstance(x[0], str):
                return x
        except Exception:
            raise ValueError(
                'Wrong input format: \'%s\'. ' % (x))

        # deal with tokenized and multiple inputs
        raise ValueError(
            '%s only supports single sentence inputs.'
            % self.__class__.__name__)

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'replace_label_ids': utils.get_placeholder(
                target, 'replace_label_ids',
                [None, self.max_seq_length], tf.int32),
            'add_label_ids': utils.get_placeholder(
                target, 'add_label_ids',
                [None, self.max_seq_length], tf.int32),
            'subtract_label_ids': utils.get_placeholder(
                target, 'subtract_label_ids',
                [None, self.max_seq_length], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = RecBERT(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            replace_label_ids=split_placeholders['replace_label_ids'],
            add_label_ids=split_placeholders['add_label_ids'],
            subtract_label_ids=split_placeholders['subtract_label_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            replace_prob=self._replace_prob,
            add_prob=self._add_prob,
            subtract_prob=self._subtract_prob,
            scope='bert',
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['replace_preds'],
               self._preds['add_preds'],
               self._preds['subtract_preds'],
               self._losses['replace_loss'],
               self._losses['add_loss'],
               self._losses['subtract_loss']]
        if as_feature:
            ops.extend([self.placeholders['input_ids'],
                        self.placeholders['replace_label_ids'],
                        self.placeholders['add_label_ids'],
                        self.placeholders['subtract_label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_inputs = output_arrays[-4]
            batch_replace_labels = output_arrays[-3]
            batch_add_labels = output_arrays[-2]
            batch_subtract_labels = output_arrays[-1]
        else:
            batch_inputs = feed_dict[self.placeholders['input_ids']]
            batch_replace_labels = \
                feed_dict[self.placeholders['replace_label_ids']]
            batch_add_labels = \
                feed_dict[self.placeholders['add_label_ids']]
            batch_subtract_labels = \
                feed_dict[self.placeholders['subtract_label_ids']]
        batch_mask = (batch_inputs != 0)

        # replace accuracy
        batch_replace_preds = output_arrays[1]
        replace_accuracy = \
            np.sum((batch_replace_preds == batch_replace_labels) \
            * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # add accuracy
        batch_add_preds = output_arrays[2]
        add_accuracy = np.sum((batch_add_preds == batch_add_labels) \
            * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # subtract accuracy
        batch_subtract_preds = output_arrays[3]
        subtract_accuracy = \
            np.sum((batch_subtract_preds == batch_subtract_labels) \
            * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # replace loss
        batch_replace_losses = output_arrays[4]
        replace_loss = np.mean(batch_replace_losses)

        # add loss
        batch_add_losses = output_arrays[5]
        add_loss = np.mean(batch_add_losses)

        # subtract loss
        batch_subtract_losses = output_arrays[6]
        subtract_loss = np.mean(batch_subtract_losses)

        info = ''
        info += ', replace_accuracy %.4f' % replace_accuracy
        info += ', add_accuracy %.4f' % add_accuracy
        info += ', subtract_accuracy %.4f' % subtract_accuracy
        info += ', replace_loss %.6f' % replace_loss
        info += ', add_loss %.6f' % add_loss
        info += ', subtract_loss %.6f' % subtract_loss

        return info

    def _get_predict_ops(self):
        return [self._preds['replace_preds'],
                self._preds['add_preds'],
                self._preds['subtract_preds']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        inputs = self.data['input_ids']
        mask = (inputs > 0)

        # replace preds
        replace_preds = []
        all_preds = utils.transform(output_arrays[0], n_inputs)
        for _preds, _mask in zip(all_preds, mask):
            input_length = np.sum(_mask)
            _pred_ids = _preds[:input_length].tolist()
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            _pred_text = utils.convert_tokens_to_text(_pred_tokens)
            replace_preds.append(_pred_text)

        # add preds
        add_preds = []
        all_preds = utils.transform(output_arrays[1], n_inputs)
        for _preds, _mask in zip(all_preds, mask):
            input_length = np.sum(_mask)
            _pred_ids = _preds[:input_length].tolist()
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            _pred_text = utils.convert_tokens_to_text(_pred_tokens)
            add_preds.append(_pred_text)

        # subtract preds
        subtract_preds = []
        all_preds = utils.transform(output_arrays[2], n_inputs)
        for _preds, _mask in zip(all_preds, mask):
            input_length = np.sum(_mask)
            _pred_ids = _preds[:input_length].tolist()
            subtract_preds.append(_pred_ids)

        outputs = {}
        outputs['replace_preds'] = replace_preds
        outputs['add_preds'] = add_preds
        outputs['subtract_preds'] = subtract_preds

        return outputs



def sample_wrong_tokens(_input_ids, _replace_label_ids,
                        _add_label_ids, _subtract_label_ids,
                        max_replace, max_add, max_subtract,
                        nonpad_seq_length, vocab_size):
    # The sampling follows the order `add -> replace -> subtract`

    # `add`, remove padding for prediction of adding tokens
    # e.g. 124 591 9521 -> 124 9521
    for _ in range(max_add):
        cand_indicies = [i for i in range(1, len(_input_ids) - 1)
                         if _input_ids[i] != 0 and
                         _input_ids[i + 1] != 0 and
                         _add_label_ids[i] == 0]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _add_label_ids[index] = _input_ids.pop(index + 1)
        _input_ids.append(0)

    # `replace`, replace tokens for prediction of replacing tokens
    # e.g. 124 591 9521 -> 124 789 9521
    for _ in range(max_replace):
        cand_indicies = [i for i in range(1, len(_input_ids) - 1)
                         if _input_ids[i] != 0 and
                         _input_ids[i] == _replace_label_ids[i]]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _input_ids[index] = random.randint(1, vocab_size - 1)

    # `subtract`, add wrong tokens for prediction of subtraction
    # e.g. 124 591 -> 124 92 591
    for _ in range(max_subtract):
        if _input_ids[-1] != 0:  # no more space
            break
        cand_indicies = [i for i in range(0, len(_input_ids) - 1)
                         if _input_ids[i] != 0 and
                         _subtract_label_ids[i] == 0]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _input_ids.insert(index, random.randint(1, vocab_size - 1))
        _subtract_label_ids[index] = 1
        _input_ids.pop()
