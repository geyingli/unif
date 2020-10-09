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

import os
import numpy as np

from uf.tools import tf
from .base import LMModule
from uf.modeling.gpt2 import GPT2
from uf.tokenization.word_piece import WordPieceTokenizer
import uf.utils as utils


class GPT2LM(LMModule):
    ''' Language modeling on GPT-2. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + 1'),
        'init_checkpoint': (
            'A string that directs to the checkpoint file used for '
            'initialization')}

    def __init__(self,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=1024,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._given = 1
        self._id_to_label = None
        self.__init_args__ = locals()

        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self.gpt2_config = get_gpt2_config(
            n_vocab=len(self.tokenizer.vocab),
            n_predict=max_seq_length,
            n_ctx=max_position_embeddings,
            n_embed=hidden_size,
            n_head=num_attention_heads,
            n_layer=num_hidden_layers)
        self._key_to_depths = get_key_to_depths(num_hidden_layers)

    def predict(self, X=None, X_tokenized=None,
                batch_size=8, given=1):
        ''' Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            given: int. The number of already known tokens.
        Returns:
            A dict object of model outputs.
        '''

        if given != self._given:
            self._given = given
            self._graph_mode = None

        return super(LMModule, self).predict(
            X, X_tokenized, batch_size)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, 'GPT2 is unsupervised. `y` should be None.'
        if '<eos>' not in self.tokenizer.vocab:
            self.tokenizer.add('<eos>')
            self.gpt2_config.n_vocab += 1
        self._eos_id = self.tokenizer.convert_tokens_to_ids(['<eos>'])[0]

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight (fit)
        if is_training:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        input_ids = []

        for ex_id, example in enumerate(X_target):
            try:
                _input_tokens = self._convert_x(example, tokenized)
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))
            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            utils.truncate_segments(
                [_input_ids], self.max_seq_length - 1,
                truncate_method=self.truncate_method)
            _input_ids.append(self._eos_id)

            if len(_input_ids) < self.max_seq_length:
                _input_ids.extend([0 for _ in range(
                    self.max_seq_length - len(_input_ids))])
            input_ids.append(_input_ids)

        return input_ids

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
            'GPT2 only supports single sentence inputs.')

    def _set_placeholders(self, target, on_export=False):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = GPT2(
            hparams=self.gpt2_config,
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='model',
            given=self._given,
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['LM'], self._losses['LM']]
        if as_feature:
            ops.extend([self.placeholders['input_ids']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_target = output_arrays[-1]
        else:
            batch_target = feed_dict[self.placeholders['input_ids']]

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
        return [self._preds['LM']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        all_preds = utils.transform(output_arrays[0], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            for i in range(self.max_seq_length):
                if _pred_tokens[i] == '<eos>':
                    _pred_tokens = _pred_tokens[:i]
                    break
            _pred_text = utils.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs['preds'] = preds

        return outputs


class GPT2Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


def get_gpt2_config(**kwargs):
    return GPT2Config(**kwargs)


def get_word_piece_tokenizer(vocab_file, do_lower_case=True):
    if not os.path.exists(vocab_file):
        raise ValueError(
            'Can\'t find vocab_file \'%s\'. '
            'Please pass the correct path of vocabulary file, '
            'e.g.`vocab.txt`.' % vocab_file)
    return WordPieceTokenizer(vocab_file, do_lower_case=do_lower_case)


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/word_embeddings': num_hidden_layers + 2,
        '/wpe': num_hidden_layers + 2,
        'ln_f/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/h%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
