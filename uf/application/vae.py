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
''' Applications based on BERT. '''

import numpy as np

from ..tools import tf
from .base import LMModule
from .bert import BERTClassifier
from ..modeling.vae import VAE
from ..tokenization.word_piece import get_word_piece_tokenizer
from .. import utils


class VAELM(BERTClassifier, LMModule):
    ''' Text generator in VAE structure. '''
    _INFER_ATTRIBUTES = {
        'max_seq_length': (
            'An integer that defines max sequence length of input tokens, '
            'which typically equals `len(tokenize(segments)) + '
            'len(segments)` + 1'),
        'init_checkpoint': (
            'A string that directs to the checkpoint file used for '
            'initialization')}

    def __init__(self,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 reduced_size=64,
                 topic_size=1024,
                 hidden_size=256,
                 num_hidden_layers=6,
                 num_attention_heads=8,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._reduced_size = reduced_size
        self._topic_size = topic_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._bias = 0
        self.__init_args__ = locals()

        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(num_hidden_layers)

        if '[SEP]' not in self.tokenizer.vocab:
            self.tokenizer.add('[SEP]')
            tf.logging.info('Add necessary token `[SEP]` into vocabulary.')

    def predict(self, X=None, X_tokenized=None,
                batch_size=8, bias=0):
        ''' Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            bias: float. The absolute value of the upper and lower range
              of random uniform noise for text generation.
        Returns:
            A dict object of model outputs.
        '''

        if bias != self._bias:
            self._bias = bias
            self._graph_mode = None

        return super(LMModule, self).predict(
            X, X_tokenized, batch_size)

    def export(self, export_dir, bias=0,
               rename_inputs=None, rename_outputs=None, ignore_outputs=None):
        ''' Export model into SavedModel files.

        Args:
            export_dir: str. Directory to which the model is saved.
            bias: float. The absolute value of the upper and lower range
              of random uniform noise for text generation.
            rename_inputs: dict. Mapping of original name to target name.
            rename_outputs: dict. Mapping of original name to target name.
            ignore_outputs: list. Name of outputs to ignore.
        Returns:
            None
        '''

        if bias != self._bias:
            self._bias = bias
            self._graph_mode = None

        return super(LMModule, self).export(
            export_dir, rename_inputs, rename_outputs, ignore_outputs)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, (
            'Training of %s is unsupervised. `y` should be None.'
            % self.__class__.__name__)

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
            _input_tokens = []
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments),
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
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = VAE(
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            reduced_size=self._reduced_size,
            topic_size=self._topic_size,
            hidden_size=self._hidden_size,
            num_hidden_layers=self._num_hidden_layers,
            num_attention_heads=self._num_attention_heads,
            bias=self._bias,
            scope='vae',
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op, self._preds['preds'], self._losses['losses']]
        if as_feature:
            ops.extend([self.placeholders['input_ids'],
                        self.placeholders['input_mask']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-2]
            batch_mask = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders['input_ids']]
            batch_mask = feed_dict[self.placeholders['input_mask']]

        # accuracy
        batch_preds = output_arrays[1]
        accuracy = (
            np.sum((batch_preds == batch_labels) * batch_mask) /
            batch_mask.sum())

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ''
        info += ', accuracy %.4f' % accuracy
        info += ', loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['miu'], self._probs['sigma'],
                self._preds['preds']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # miu
        miu = utils.transform(output_arrays[0], n_inputs)

        # sigma
        sigma = utils.transform(output_arrays[1], n_inputs)

        # preds
        all_preds = utils.transform(output_arrays[2], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            for i in range(self.max_seq_length):
                if _pred_ids[i] == 0:
                    _pred_tokens = _pred_tokens[:i]
                    break
            preds.append(_pred_tokens)

        outputs = {}
        outputs['miu'] = miu
        outputs['sigma'] = sigma
        outputs['preds'] = preds

        return outputs


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 3,
        '/encoder/projection': 2,
        '/decoder': 1,
        'cls/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/layer_%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 2
    return key_to_depths
