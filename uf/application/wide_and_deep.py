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
''' Applications based on Wide & Deep model. '''

import copy
import numpy as np

from uf.tools import tf
from .base import ClassifierModule
from .bert import BERTClassifier, get_bert_config
from .albert import get_albert_config
from uf.modeling.bert import BERTEncoder
from uf.modeling.albert import ALBERTEncoder
from uf.modeling.wide_and_deep import WideAndDeepDecoder
from uf.tokenization.word_piece import get_word_piece_tokenizer
import uf.utils as utils



class WideAndDeepClassifier(BERTClassifier, ClassifierModule):
    ''' Single-label classifier on Wide & Deep model with BERT. '''
    _INFER_ATTRIBUTES = copy.deepcopy(BERTClassifier._INFER_ATTRIBUTES)
    _INFER_ATTRIBUTES['wide_features'] = \
        'A list of possible values for `Wide` features (integer or string)'

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 wide_features=None,
                 deep_module='bert',
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self.wide_features = wide_features
        self._deep_module = deep_module
        self._id_to_label = None
        self.__init_args__ = locals()

        if deep_module == 'albert':
            self.bert_config = get_albert_config(config_file)
        else:
            self.bert_config = get_bert_config(config_file)

        assert deep_module in ('bert', 'roberta', 'albert', 'electra'), (
            'Invalid value of `deep_module`: %s. Pick one from '
            '`bert`, `roberta`, `albert` and `electra`.')
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
            (input_ids, input_mask, segment_ids,
             n_wide_features, wide_features) = self._convert_X(
                 X_tokenized if tokenized else X, tokenized=tokenized)
            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            data['n_wide_features'] = np.array(n_wide_features, dtype=np.int32)
            data['wide_features'] = np.array(wide_features, dtype=np.int32)
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
        segment_inputs = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_inputs.append(
                    {'Wide': example['Wide'],
                     'Deep': self._convert_x(example['Deep'], tokenized)})
            except Exception:
                raise ValueError(
                    'Wrong input format (line %d): \'%s\'. An untokenized '
                    'example: X = [{\'Wide\': [1, 5, \'positive\'], '
                    '\'Deep\': \'I bet she will win.\'}, ...]'
                    % (ex_id, example))

        if self.wide_features is None:
            self.wide_features = set()
            for segments in segment_inputs:
                for feature in segments['Wide']:
                    self.wide_features.add(feature)
            self.wide_features = list(self.wide_features)
        elif not isinstance(self.wide_features, list):
            raise ValueError(
                '`wide_features` should be a list of possible values '
                '(integer or string). '
                'E.g. [1, \'Positive\', \'Subjective\'].')
        wide_features_map = {
            self.wide_features[i]: i + 1
            for i in range(len(self.wide_features))}

        input_ids = []
        input_mask = []
        segment_ids = []
        n_wide_features = []
        wide_features = []
        for ex_id, segments in enumerate(segment_inputs):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _wide_features = []
            for feature in segments['Wide']:
                try:
                    _wide_features.append(wide_features_map[feature])
                except:
                    tf.logging.warning(
                        'Unregistered wide feature: %s. Ignored.' % feature)
                    continue
            _n_wide_features = len(_wide_features)

            segments = segments['Deep']
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
            for _ in range(len(self.wide_features) - _n_wide_features):
                _wide_features.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            n_wide_features.append(_n_wide_features)
            wide_features.append(_wide_features)

        return (input_ids, input_mask, segment_ids,
                n_wide_features, wide_features)

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
            'n_wide_features': utils.get_placeholder(
                target, 'n_wide_features',
                [None], tf.int32),
            'wide_features': utils.get_placeholder(
                target, 'wide_features',
                [None, len(self.wide_features)], tf.int32),
            'label_ids': utils.get_placeholder(
                target, 'label_ids', [None], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        def _get_encoder(model_name):
            if model_name == 'bert' or model_name == 'roberta':
                encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='bert',
                    **kwargs)
            elif model_name == 'albert':
                encoder = ALBERTEncoder(
                    albert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='bert',
                    **kwargs)
            elif model_name == 'electra':
                encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders['input_ids'],
                    input_mask=split_placeholders['input_mask'],
                    segment_ids=split_placeholders['segment_ids'],
                    scope='electra',
                    **kwargs)
            return encoder

        encoder = _get_encoder(self._deep_module)
        encoder_output = encoder.get_pooled_output()
        decoder = WideAndDeepDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            n_wide_features=split_placeholders['n_wide_features'],
            wide_features=split_placeholders['wide_features'],
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/seq_relationship',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': num_hidden_layers + 2,
        'wide/': 2,
        'wide_and_deep/': 1,
        '/pooler/': 1,
        'cls/': 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths['/layer_%d/' % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
