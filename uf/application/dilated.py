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
''' Applications based on GPT-2. '''

import random
import numpy as np

from uf.tools import tf
from .base import LMModule
from .bert import get_bert_config, get_word_piece_tokenizer, get_key_to_depths
from uf.modeling.dilated import DLM
import uf.utils as utils


class DilatedLM(LMModule):
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
                 dupe_factor=1,
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
        self._dupe_factor = dupe_factor
        self._replace_prob = replace_prob
        self._add_prob = add_prob
        self._subtract_prob = subtract_prob
        self._loop = 1
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

    def predict(self, X=None, X_tokenized=None,
                batch_size=8, loop=1):

        if loop != self._loop:
            self._loop = loop
            self._graph_mode = None

        return super(LMModule, self).predict(
            X, X_tokenized, batch_size)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, ('%s is unsupervised. `y` should be None.'
                           % self.__class__.__name__)
        if '<spad>' not in self.tokenizer.vocab:
            self.tokenizer.add('<spad>')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            (input_ids, input_mask, label_ids) = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized,
                is_training=is_training)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)

            if is_training:
                data['label_ids'] = np.array(label_ids, dtype=np.int32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight (fit)
        if is_training:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized, is_training):
        input_ids = []
        input_mask = []
        label_ids = []

        dupe_factor = self._dupe_factor if is_training else 1
        for _ in range(dupe_factor):

            for ex_id, example in enumerate(X_target):
                try:
                    _input_tokens = self._convert_x(example, tokenized)
                except Exception:
                    tf.logging.warning(
                        'Wrong input format (line %d): \'%s\'. '
                        % (ex_id, example))
                _input_tokens = ['[CLS]'] + _input_tokens
                _input_ids = self.tokenizer.convert_tokens_to_ids(
                    _input_tokens)

                utils.truncate_segments(
                    [_input_ids], self.max_seq_length,
                    truncate_method=self.truncate_method)
                nonpad_seq_length = len(_input_ids)
                _input_mask = [1] * nonpad_seq_length

                if nonpad_seq_length < self.max_seq_length:
                    _input_ids.extend(
                        [0] * (self.max_seq_length - nonpad_seq_length))
                    _input_mask.extend(
                        [0] * (self.max_seq_length - nonpad_seq_length))

                _dilated_ids = []
                _dilated_mask = []
                _label_ids = []
                for i, _input_id in enumerate(_input_ids):
                    _dilated_ids.extend([_input_id, 0])
                    _dilated_mask.extend([_input_mask[i], _input_mask[i]])
                    _label_ids.extend([_input_id, 0])

                # replace/add/subtract
                if is_training:
                    max_replace = int(nonpad_seq_length * self._replace_prob)
                    max_add = int(nonpad_seq_length * self._add_prob)
                    max_subtract = int(nonpad_seq_length * self._subtract_prob)

                    sample_wrong_tokens(
                        _dilated_ids, _dilated_mask, _label_ids,
                        max_replace, max_add, max_subtract,
                        nonpad_seq_length=nonpad_seq_length,
                        vocab_size=len(self.tokenizer.vocab))

                input_ids.append(_dilated_ids)
                input_mask.append(_dilated_mask)
                label_ids.append(_label_ids)

        return input_ids, input_mask, label_ids

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
            '%s only supports single sentence inputs.'
            % self.__class__.__name__)

    def _set_placeholders(self, target, on_export=False):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length * 2], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length * 2], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids',
                [None, self.max_seq_length * 2], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = DLM(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            max_seq_length=self.max_seq_length,
            spad_id=self.tokenizer.convert_tokens_to_ids(['<spad>'])[0],
            loop=self._loop,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='dilated',
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['LM'], self._losses['LM']]
        if as_feature:
            ops.extend([self.placeholders['input_ids'],
                        self.placeholders['label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_inputs = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_inputs = feed_dict[self.placeholders['input_ids']]
            batch_labels = feed_dict[self.placeholders['label_ids']]

        # accuracy
        batch_preds = output_arrays[1]
        batch_mask = (batch_inputs != batch_labels)
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
        return [self._preds['LM']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        all_preds = utils.transform(output_arrays[0], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_ids = [_pred_id for _pred_id in _pred_ids[1:]
                         if _pred_id != 0]
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            _pred_text = utils.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs['preds'] = preds

        return outputs


def sample_wrong_tokens(_dilated_ids, _dilated_mask, _label_ids,
                        max_replace, max_add, max_subtract,
                        nonpad_seq_length, vocab_size):

    # The sampling follows the order `add -> replace -> subtract`

    # `add`, remove padding for prediction of adding tokens
    # e.g. 124 0 591 0 9521 -> 124 591 9521 0 0
    for _ in range(max_add):
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] != 0 and
                         _dilated_ids[i - 1] == 0 and
                         _dilated_ids[i + 1] == 0]
        if not cand_indicies:
            break

        def mod_add(list_obj, index):
            list_obj.pop(index + 1)
            list_obj.pop(index - 1)
            list_obj.extend([0, 0])
        index = random.choice(cand_indicies)
        mod_add(_dilated_ids, index)
        mod_add(_dilated_mask, index)
        mod_add(_label_ids, index)
        _dilated_ids[index - 1] = 0

    # `replace`, replace tokens for prediction of replacing tokens
    # e.g. 124 0 591 0 9521 -> 124 0 789 0 9521
    for _ in range(max_replace):
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] != 0 and
                         _dilated_ids[i - 1] == 0 and
                         _dilated_ids[i + 1] == 0 and
                         _dilated_ids[i] == _label_ids[i]]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _dilated_ids[index] = random.randint(1, vocab_size - 1)

    # `subtract`, add wrong tokens for prediction of subtraction
    # e.g. 124 0 591 0 9521 -> 124 0 92 0 591
    for _ in range(max_subtract):
        if _dilated_mask[-2] == 1:
            break
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] == 0 and
                         _dilated_ids[i - 1] != 0 and
                         _dilated_ids[i + 1] != 0 and
                         _dilated_ids[i - 1] == _label_ids[i - 1] and
                         _dilated_ids[i + 1] == _label_ids[i + 1]]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _dilated_ids.insert(index, random.randint(1, vocab_size - 1))
        _dilated_ids.insert(index, 0)
        _dilated_ids.pop()
        _dilated_ids.pop()
        _dilated_mask.insert(index, 1)
        _dilated_mask.insert(index, 1)
        _dilated_mask.pop()
        _dilated_mask.pop()
        _label_ids.insert(index, 0)
        _label_ids.insert(index, 0)
        _label_ids.pop()
        _label_ids.pop()

