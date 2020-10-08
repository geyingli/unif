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
''' Applications based on FastBERT. '''

import numpy as np

from .base import ClassifierModule
from uf.modeling.fast_bert import FastBERTCLSDistillor
from .bert import BERTClassifier, get_bert_config, get_word_piece_tokenizer
import uf.utils as utils


class FastBERTClassifier(BERTClassifier, ClassifierModule):
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 cls_model='self-attention',
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self._cls_model = cls_model
        self._ignore_cls = []
        self._speed = 0.1
        self._drop_pooler = drop_pooler
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = 'unsupported'

    def predict(self, X=None, X_tokenized=None,
                batch_size=8, speed=0.1, ignore_cls=None):

        if ignore_cls != self._ignore_cls:
            if not ignore_cls:
                ignore_cls = self._ignore_cls
            elif isinstance(ignore_cls, str):
                ignore_cls = ignore_cls.replace(' ','').split(',')
            elif isinstance(ignore_cls, list):
                pass
            else:
                raise ValueError(
                    '`ignore_cls` should be a list of child-classifier ids or '
                    'a string seperated with commas.')
            self._ignore_cls = ignore_cls
            self._graph_mode = None

        if speed != self._speed:
            if not speed:
                raise ValueError(
                    '`speed` should be a float number between `0` and `1`.')
            self._speed = speed
            self._graph_mode = None

        return super(FastBERTClassifier, self).predict(
            X, X_tokenized, batch_size)

    def score(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                batch_size=8, speed=0.1, ignore_cls=None):

        if ignore_cls != self._ignore_cls:
            if not ignore_cls:
                ignore_cls = self._ignore_cls
            elif isinstance(ignore_cls, str):
                ignore_cls = ignore_cls.replace(' ','').split(',')
            elif isinstance(ignore_cls, list):
                pass
            else:
                raise ValueError(
                    '`ignore_cls` should be a list of child-classifier ids or '
                    'a string seperated with commas.')
            self._ignore_cls = ignore_cls
            self._graph_mode = None

        if speed != self._speed:
            if not speed:
                raise ValueError(
                    '`speed` should be a float number between `0` and `1`.')
            self._speed = speed
            self._graph_mode = None

        return super(FastBERTClassifier, self).score(
            X, y, sample_weight, X_tokenized, batch_size)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert not y, (
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

        if y:
            # convert y and sample_weight
            label_ids = self._convert_y(y, n_inputs)
            data['label_ids'] = np.array(label_ids, dtype=np.int32)

            # convert sample_weight (fit, score)
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _forward(self, is_training, split_placeholders, **kwargs):

        distillor = FastBERTCLSDistillor(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='bert',
            name='cls',
            drop_pooler=self._drop_pooler,
            speed=self._speed,
            ignore_cls=[] if is_training else self._ignore_cls,
            cls_model=self._cls_model,
            label_size=self.label_size,
            **kwargs)
        (total_loss, losses, probs, preds) = distillor.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        return [self._train_op, self._losses['distill']]

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ''
        info += ', distill loss %.6f' % loss

        return info

    def _get_predict_ops(self):
        return [self._probs['cls']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        def _permutate(batch_probs):
            n_device = max(len(self._gpu_ids), 1)
            d_batch_size = self.batch_size // n_device
            probs = np.zeros((self.batch_size, self.label_size))
            sources = np.zeros((self.batch_size), dtype=np.int32)
            max_loop = \
                self.bert_config.num_hidden_layers + 1 - len(self._ignore_cls)
            keep_cls = [
                cls_idx for cls_idx \
                in list(range(self.bert_config.num_hidden_layers + 1)) \
                if cls_idx not in self._ignore_cls]
            i = 0
            _uncertainty = lambda prob: (prob * np.log(prob) +
                                         (1 - prob) * np.log(1 - prob)) / \
                                        np.log(1 / self.label_size)

            for d in range(n_device):
                unfinished = [k + i for k in range(d_batch_size)]

                for loop in range(max_loop):
                    source = keep_cls[loop]
                    next_unfinished = []

                    for k in range(len(unfinished)):
                        if _uncertainty(batch_probs[i][0]) < self._speed or \
                                loop == max_loop - 1:
                            probs[unfinished[k]] = batch_probs[i]
                            sources[unfinished[k]] = source
                        else:
                            next_unfinished.append(unfinished[k])
                        i += 1
                    unfinished = next_unfinished
            assert i == len(batch_probs)
            return probs, sources

        # probs
        probs_arrays = []
        sources_arrays = []
        for batch_probs in output_arrays[0]:
            probs_array, sources_array = _permutate(batch_probs)
            probs_arrays.append(probs_array)
            sources_arrays.append(sources_array)
        probs = utils.transform(probs_arrays, n_inputs)
        sources = utils.transform(sources_arrays, n_inputs).tolist()

        # preds
        preds = np.argmax(probs, axis=-1).tolist()
        if self._id_to_label:
            preds = [self._id_to_label[idx] for idx in preds]

        outputs = {}
        outputs['preds'] = preds
        outputs['probs'] = probs
        outputs['sources'] = sources

        return outputs

    def _get_score_ops(self):
        return [self._probs['cls']]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        def _permutate(batch_probs):
            n_device = max(len(self._gpu_ids), 1)
            d_batch_size = self.batch_size // n_device
            probs = np.zeros((self.batch_size, self.label_size))
            sources = np.zeros((self.batch_size), dtype=np.int32)
            max_loop = \
                self.bert_config.num_hidden_layers + 1 - len(self._ignore_cls)
            keep_cls = [
                cls_idx for cls_idx \
                in list(range(self.bert_config.num_hidden_layers + 1)) \
                if cls_idx not in self._ignore_cls]
            i = 0
            _uncertainty = lambda prob: (prob * np.log(prob) +
                                         (1 - prob) * np.log(1 - prob)) / \
                                        np.log(1 / self.label_size)

            for d in range(n_device):
                unfinished = [k + i for k in range(d_batch_size)]

                for loop in range(max_loop):
                    source = keep_cls[loop]
                    next_unfinished = []

                    for k in range(len(unfinished)):
                        if _uncertainty(batch_probs[i][0]) < self._speed or \
                                loop == max_loop - 1:
                            probs[unfinished[k]] = batch_probs[i]
                            sources[unfinished[k]] = source
                        else:
                            next_unfinished.append(unfinished[k])
                        i += 1
                    unfinished = next_unfinished
            assert i == len(batch_probs)
            return probs, sources

        def _transform(output_arrays):
            if len(output_arrays[0].shape) > 1:
                return np.vstack(output_arrays)[:n_inputs]
            return np.hstack(output_arrays)[:n_inputs]

        # accuracy
        probs_arrays = []
        for batch_probs in output_arrays[0]:
            probs_array, _ = _permutate(batch_probs)
            probs_arrays.append(probs_array)
        probs = _transform(probs_arrays)
        preds = np.argmax(probs, axis=-1)
        labels = self.data['label_ids']
        accuracy = np.mean(preds == labels)

        # loss
        losses = [-np.log(probs[i][label]) for i, label in enumerate(labels)]
        sample_weight = self.data['sample_weight']
        losses = np.array(losses) * sample_weight
        loss = np.mean(losses)

        outputs = {}
        outputs['accuracy'] = accuracy
        outputs['loss'] = loss

        return outputs
