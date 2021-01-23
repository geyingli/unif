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
                 add_prob=0.1,
                 del_prob=0.1,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._on_predict = False
        self._add_prob = add_prob
        self._del_prob = del_prob
        self.__init_args__ = locals()

        assert add_prob <= 0.5, (
            'The value of `add_prob` should be larger than 0 '
            'and smaller than 1/2.')
        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

    def predict(self, X=None, X_tokenized=None,
                batch_size=8):
        ''' Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
        Returns:
            A dict object of model outputs.
        '''

        self._on_predict = True
        ret = super(LMModule, self).predict(
            X, X_tokenized, batch_size)
        self._on_predict = False

        return ret

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
            X_target = X_tokenized if tokenized else X
            (input_ids, add_label_ids, del_label_ids) = self._convert_X(
                X_target, tokenized=tokenized,
                is_training=is_training)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)

            if is_training:
                data['add_label_ids'] = np.array(
                    add_label_ids, dtype=np.int32)
                data['del_label_ids'] = np.array(
                    del_label_ids, dtype=np.int32)

            # backup for answer mapping
            if self._on_predict:
                self._tokenized = tokenized
                self._X_target = X_target

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

        # backup for answer mapping
        if self._on_predict:
            self._input_tokens = []

        # tokenize input texts and scan over corpus
        tokenized_input_ids = []
        vocab_size = len(self.tokenizer.vocab)
        vocab_ind = list(range(vocab_size))
        vocab_p = [0] * vocab_size
        for ex_id, sample in enumerate(X_target):
            _input_tokens = self._convert_x(sample, tokenized)

            # skip noise training data
            if is_training:
                if len(_input_tokens) > self.max_seq_length:
                    continue
            else:
                utils.truncate_segments(
                    [_input_tokens], self.max_seq_length,
                    truncate_method=self.truncate_method)

            # backup for answer mapping
            if self._on_predict:
                self._input_tokens.append(_input_tokens)

            # count char
            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            if is_training:
                for _input_id in _input_ids:
                    vocab_p[_input_id] += 1

            tokenized_input_ids.append(_input_ids)
        if is_training:
            vocab_p_sum = sum(vocab_p)
            vocab_p = [n / vocab_p_sum for n in vocab_p]

        input_ids = []
        add_label_ids = []
        del_label_ids = []
        for ex_id in range(len(tokenized_input_ids)):
            _input_ids = tokenized_input_ids[ex_id]

            nonpad_seq_length = len(_input_ids)
            for _ in range(self.max_seq_length - nonpad_seq_length):
                _input_ids.append(0)

            _add_label_ids = []
            _del_label_ids = []

            # add/del
            if is_training:
                if (ex_id + 1) % 100000 == 0:
                    tf.logging.info(
                        'Sampling wrong tokens of input %d' % (ex_id + 1))

                for _ in range(self.max_seq_length):
                    _add_label_ids.append(0)
                    _del_label_ids.append(0)

                max_add = 0
                max_del = 0
                for _ in range(nonpad_seq_length):
                    if random.random() < self._add_prob:
                        max_add += 1
                    if random.random() < self._del_prob:
                        max_del += 1

                sample_wrong_tokens(
                    _input_ids, _add_label_ids, _del_label_ids,
                    max_add=max_add, max_del=max_del,
                    vocab_size=vocab_size,
                    vocab_ind=vocab_ind,
                    vocab_p=vocab_p)

            input_ids.append(_input_ids)
            add_label_ids.append(_add_label_ids)
            del_label_ids.append(_del_label_ids)

        return input_ids, add_label_ids, del_label_ids

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
            'add_label_ids': utils.get_placeholder(
                target, 'add_label_ids',
                [None, self.max_seq_length], tf.int32),
            'del_label_ids': utils.get_placeholder(
                target, 'del_label_ids',
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
            add_label_ids=split_placeholders['add_label_ids'],
            del_label_ids=split_placeholders['del_label_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            add_prob=self._add_prob,
            del_prob=self._del_prob,
            scope='bert',
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['add_preds'],
               self._preds['del_preds'],
               self._losses['add_loss'],
               self._losses['del_loss']]
        if as_feature:
            ops.extend([self.placeholders['input_ids'],
                        self.placeholders['add_label_ids'],
                        self.placeholders['del_label_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_inputs = output_arrays[-3]
            batch_add_labels = output_arrays[-2]
            batch_del_labels = output_arrays[-1]
        else:
            batch_inputs = feed_dict[self.placeholders['input_ids']]
            batch_add_labels = \
                feed_dict[self.placeholders['add_label_ids']]
            batch_del_labels = \
                feed_dict[self.placeholders['del_label_ids']]
        batch_mask = (batch_inputs != 0)

        # add accuracy
        batch_add_preds = output_arrays[1]
        add_accuracy = np.sum((batch_add_preds == batch_add_labels) \
            * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # del accuracy
        batch_del_preds = output_arrays[2]
        del_accuracy = \
            np.sum((batch_del_preds == batch_del_labels) \
            * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # add loss
        batch_add_losses = output_arrays[3]
        add_loss = np.mean(batch_add_losses)

        # del loss
        batch_del_losses = output_arrays[4]
        del_loss = np.mean(batch_del_losses)

        info = ''
        if self._add_prob > 0:
            info += ', add_accuracy %.4f' % add_accuracy
            info += ', add_loss %.6f' % add_loss
        if self._del_prob > 0:
            info += ', del_accuracy %.4f' % del_accuracy
            info += ', del_loss %.6f' % del_loss

        return info

    def _get_predict_ops(self):
        return [self._preds['add_preds'],
                self._preds['del_preds']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        input_ids = self.data['input_ids']
        mask = (input_ids > 0)

        # integrated preds
        preds = []
        add_preds = utils.transform(output_arrays[0], n_inputs)
        del_preds = utils.transform(output_arrays[1], n_inputs)
        for ex_id in range(n_inputs):
            _add_preds = add_preds[ex_id]
            _del_preds = del_preds[ex_id]
            _input_length = np.sum(mask[ex_id])
            _input_tokens = self._input_tokens[ex_id]
            _output_tokens = [token for token in _input_tokens]

            if self._tokenized:
                n = 0
                for i in range(_input_length):
                    if self._del_prob > 0 and _del_preds[i] != 0:
                        _token = '{del:%s}' % _output_tokens[i + n]
                        _output_tokens[i + n] = _token
                    if self._add_prob > 0 and _add_preds[i] != 0:
                        _token = self.tokenizer.convert_ids_to_tokens(
                            [_add_preds[i]])[0]
                        _token = '{add:%s}' % _token
                        _output_tokens.insert(i + 1 + n, _token)
                        n += 1
                preds.append(_output_tokens)
            else:
                _text = self._X_target[ex_id]
                _mapping_start, _mapping_end = utils.align_tokens_with_text(
                    _input_tokens, _text, self._do_lower_case)

                n = 0
                for i in range(_input_length):
                    if self._del_prob > 0 and _del_preds[i] != 0:
                        _start_ptr = _mapping_start[i] + n
                        _end_ptr = _mapping_end[i] + n
                        _del_token = _text[_start_ptr: _end_ptr]

                        _token = '{del:%s}' % _del_token
                        _text = _text[:_start_ptr] + _token + _text[_end_ptr:]
                        n += len(_token) - len(_del_token)
                    if self._add_prob > 0 and _add_preds[i] != 0:
                        _token = self.tokenizer.convert_ids_to_tokens(
                            [_add_preds[i]])[0]
                        _token = '{add:%s}' % _token
                        _ptr = _mapping_end[i] + n
                        _text = _text[:_ptr] + _token + _text[_ptr:]
                        n += len(_token)
                preds.append(_text)

        outputs = {}
        outputs['preds'] = preds

        return outputs



def sample_wrong_tokens(_input_ids, _add_label_ids, _del_label_ids,
                        max_add, max_del, vocab_size, vocab_ind, vocab_p):

    # `add`, remove padding for prediction of adding tokens
    # e.g. 124 591 9521 -> 124 9521
    for _ in range(max_add):
        cand_indicies = [i for i in range(0, len(_input_ids) - 1)
                         if _input_ids[i + 1] != 0 and
                         _add_label_ids[i] == 0 and
                         _add_label_ids[i + 1] == 0]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _add_label_ids[index] = _input_ids.pop(index + 1)
        _add_label_ids.pop(index + 1)
        _del_label_ids.pop(index + 1)
        _input_ids.append(0)
        _add_label_ids.append(0)
        _del_label_ids.append(0)

    # `del`, add wrong tokens for prediction of deleted tokens
    # e.g. 124 591 -> 124 92 591
    for _ in range(max_del):
        if _input_ids[-1] != 0:  # no more space
            break
        nonpad_seq_length = \
            sum([1 if _token != 0 else 0 for _token in _input_ids])
        cand_indicies = list(range(nonpad_seq_length + 1))

        index = random.choice(cand_indicies)
        rand = np.random.choice(vocab_ind, p=vocab_p)  # sample from vocabulary

        # always delete the right one
        _input_ids.insert(index, rand)
        _add_label_ids.insert(index, 0)
        _del_label_ids.insert(index, 1)
        if rand == _input_ids[index + 1]:
            _del_label_ids[index + 1], _del_label_ids[index] = \
                _del_label_ids[index], _del_label_ids[index + 1]

        _input_ids.pop()
        _add_label_ids.pop()
        _del_label_ids.pop()
