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
''' Base class for APIs with general methods, typically evaluation metrics. '''

import numpy as np

from ..core import BaseModule
from .. import utils


class ClassifierModule(BaseModule):
    ''' Application class of classification. '''
    pass


class NERModule(BaseModule):
    ''' Application class of name entity recognition (NER). '''

    O_ID = 0
    B_ID = 1
    I_ID = 2
    E_ID = 3
    S_ID = 4

    def _get_cascade_f1(self, preds, labels, mask):
        metrics = {}

        # f1 of each type
        B_ids = []
        for i, entity_type in enumerate(self.entity_types):
            B_id = 1 + i * 4
            B_ids.append(B_id)
            f1_token, f1_entity = self._get_f1(
                preds, labels, mask, B_id=B_id)
            metrics['f1/%s-token' % entity_type] = f1_token
            metrics['f1/%s-entity' % entity_type] = f1_entity

        # macro f1
        f1_macro_token = np.mean(
            [metrics[key] for key in metrics if '-token' in key])
        f1_macro_entity = np.mean(
            [metrics[key] for key in metrics if '-entity' in key])
        metrics['macro f1/token'] = f1_macro_token
        metrics['macro f1/entity'] = f1_macro_entity

        # micro f1
        f1_micro_token, f1_micro_entity = self._get_f1(
            preds, labels, mask, B_id=B_ids)
        metrics['micro f1/token'] = f1_micro_token
        metrics['micro f1/entity'] = f1_micro_entity

        return metrics

    def _get_f1(self, preds, labels, mask, B_id=None):

        tp_token, fp_token, fn_token = 0, 0, 0
        tp_entity, fp_entity, fn_entity = 0, 0, 0
        for _preds, _labels, _mask in zip(preds, labels, mask):
            length = int(np.sum(_mask))

            # entity metrics
            _entities_pred = self._get_entities(_preds[:length], B_id)
            _entities_label = self._get_entities(_labels[:length], B_id)
            for _entity_pred in _entities_pred:
                if _entity_pred in _entities_label:
                    tp_entity += 1
                else:
                    fp_entity += 1
            for _entity_label in _entities_label:
                if _entity_label not in _entities_pred:
                    fn_entity += 1

            # token metrics
            _preds = np.zeros((length))
            _labels = np.zeros((length))
            for _entity in _entities_pred:
                _preds[_entity[0]:_entity[1] + 1] = 1
            for _entity in _entities_label:
                _labels[_entity[0]:_entity[1] + 1] = 1
            for _pred_id, _label_id in zip(_preds, _labels):
                if _pred_id == 1:
                    if _label_id == 1:
                        tp_token += 1
                    else:
                        fp_token += 1
                elif _label_id == 1:
                    fn_token += 1

        pre_token = tp_token / (tp_token + fp_token + 1e-6)
        rec_token = tp_token / (tp_token + fn_token + 1e-6)
        f1_token = 2 * pre_token * rec_token / (
            pre_token + rec_token + 1e-6)

        pre_entity = tp_entity / (tp_entity + fp_entity + 1e-6)
        rec_entity = tp_entity / (tp_entity + fn_entity + 1e-6)
        f1_entity = 2 * pre_entity * rec_entity / (
            pre_entity + rec_entity + 1e-6)

        return f1_token, f1_entity

    def _get_entities(self, ids, B_id=None):
        if not B_id:
            B_id = self.B_ID
            I_id = self.I_ID
            E_id = self.E_ID
            S_id = self.S_ID
            BIES = [(B_id, I_id, E_id, S_id)]
        elif isinstance(B_id, list):
            BIES = []
            for _B_id in B_id:
                I_id = _B_id + 1
                E_id = _B_id + 2
                S_id = _B_id + 3
                BIES += [(_B_id, I_id, E_id, S_id)]
        else:
            I_id = B_id + 1
            E_id = B_id + 2
            S_id = B_id + 3
            BIES = [(B_id, I_id, E_id, S_id)]

        entities = []
        for k in range(len(BIES)):
            (B_id, I_id, E_id, S_id) = BIES[k]
            on_entity = False
            start_id = 0
            for i in range(len(ids)):
                if on_entity:
                    if ids[i] == I_id:
                        pass
                    elif ids[i] == E_id:
                        on_entity = False
                        entities.append([start_id, i])
                    elif ids[i] == S_id:
                        on_entity = False
                        entities.append([i, i])
                    else:
                        on_entity = False
                else:
                    if ids[i] == B_id:
                        on_entity = True
                        start_id = i
                    elif ids[i] == S_id:
                        entities.append([i, i])
        return entities


class MRCModule(BaseModule):
    ''' Application class of machine reading comprehension (MRC). '''

    def _get_em_and_f1(self, preds, labels):
        em, f1 = 0, 0
        for _preds, _labels in zip(preds, labels):
            start_pred, end_pred = int(_preds[0]), int(_preds[1])
            start_label, end_label = int(_labels[0]), int(_labels[1])

            # no answer prediction
            if start_pred == 0 or end_pred == 0 or start_pred > end_pred:
                if start_label == 0:
                    em += 1
                    f1 += 1

            # answer prediction (no intersection)
            elif start_pred > end_label or end_pred < start_label:
                pass

            # answer prediction (has intersection)
            else:
                tp = (min(end_pred, end_label) + 1 -
                      max(start_pred, start_label))
                fp = (max(0, end_pred - end_label) + max(0, start_label - start_pred))
                fn = (max(0, start_pred - start_label) + max(0, end_label - end_pred))
                if fp + fn == 0:
                    em += 1
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 += 2 * precision * recall / (precision + recall + 1e-6)

        em /= len(labels)
        f1 /= len(labels)
        return em, f1


class MTModule(BaseModule):
    ''' Application class of machine translation (MT). '''

    def _get_bleu(self, preds, labels, mask, max_gram=4):
        eos_id = self.tokenizer.convert_tokens_to_ids(['</s>'])[0]

        bleus = []
        for _preds, _labels, _mask in zip(preds, labels, mask):

            # preprocess
            for i in range(len(_preds)):
                if _preds[i] == eos_id:
                    _preds = _preds[:i+1]
                    break
            _labels = _labels[:int(np.sum(_mask)) - 1]  # remove </s>

            power = 0
            for n in range(max_gram):
                ngrams = []
                nominator = 0
                denominator = 0

                for i in range(len(_labels) - n):
                    ngram = _labels[i:i+1+n].tolist()
                    if ngram in ngrams:
                        continue
                    cand_count = len(utils.find_all_boyer_moore(_preds, ngram))
                    ref_count = len(utils.find_all_boyer_moore(_labels, ngram))
                    nominator += min(cand_count, ref_count)
                    denominator += cand_count
                    ngrams.append(ngram)

                power += 1 / (n + 1) * np.log(
                    nominator / (denominator + 1e-6) + 1e-6)

            _bleu = np.exp(power)
            if len(_preds) >= len(_labels):
                _bleu *= np.exp(1 - len(_labels) / len(_preds))
            bleus.append(_bleu)

        return np.mean(bleus)

    def _get_rouge(self, preds, labels, mask, max_gram=4):
        eos_id = self.tokenizer.convert_tokens_to_ids(['</s>'])[0]

        rouges = []
        for _preds, _labels, _mask in zip(preds, labels, mask):

            # preprocess
            for i in range(len(_preds)):
                if _preds[i] == eos_id:
                    _preds = _preds[:i+1]
                    break
            _labels = _labels[:int(np.sum(_mask)) - 1]  # remove </s>

            nominator = 0
            denominator = 0
            for n in range(max_gram):
                ngrams = []

                for i in range(len(_labels) - n):
                    ngram = _labels[i:i+1+n].tolist()
                    if ngram in ngrams:
                        continue
                    nominator += len(utils.find_all_boyer_moore(_preds, ngram))
                    denominator += \
                        len(utils.find_all_boyer_moore(_labels, ngram))
                    ngrams.append(ngram)

            _rouge = nominator / (denominator + 1e-6)
            rouges.append(_rouge)

        return np.mean(rouges)


class LMModule(BaseModule):
    ''' Application class of language modeling (LM). '''

    def fit_from_tfrecords(
            self, batch_size=32,
            learning_rate=5e-5,
            target_steps=None,
            total_steps=1000000,
            warmup_ratio=0.01,
            print_per_secs=0.1,
            save_per_steps=10000,
            tfrecords_files=None,
            n_jobs=3,
            **kwargs):
        ''' Training the model using TFRecords.

        Args:
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            tfrecords_files: list. A list object of string defining TFRecords
              files to read.
            n_jobs: int. Number of threads in reading TFRecords files.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        '''
        super().fit_from_tfrecords(
            batch_size,
            learning_rate,
            target_steps,
            total_steps,
            warmup_ratio,
            print_per_secs,
            save_per_steps,
            tfrecords_files,
            n_jobs,
            **kwargs)

    def fit(self, X=None, y=None, sample_weight=None, X_tokenized=None,
            batch_size=32,
            learning_rate=5e-5,
            target_steps=None,
            total_steps=1000000,
            warmup_ratio=0.01,
            print_per_secs=0.1,
            save_per_steps=10000,
            **kwargs):
        ''' Training the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            y: list. A list object consisting labels.
            sample_weight: list. A list object of float-convertable values.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        '''
        super().fit(
            X, y, sample_weight, X_tokenized,
            batch_size,
            learning_rate,
            target_steps,
            total_steps,
            warmup_ratio,
            print_per_secs,
            save_per_steps,
            **kwargs)

    def score(self, *args, **kwargs):
        raise AttributeError(
            '`score` method is not supported for unsupervised '
            'language modeling (LM) modules.')
