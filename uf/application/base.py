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
''' Base class for APIs with general methods, typically evaluation metrics. '''

import numpy as np

from uf.core import BaseModule
import uf.utils as utils


class ClassifierModule(BaseModule):
    pass


class NERModule(BaseModule):

    O_ID = 0
    B_ID = 1
    I_ID = 2
    E_ID = 3
    S_ID = 4

    def _get_f1(self, preds, labels, mask, B_id=None):

        tp_token, fp_token, fn_token = 0, 0, 0
        tp_entity, fp_entity, fn_entity = 0, 0, 0
        for _preds, _labels, _mask in zip(
                preds, labels, mask):
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
        f1_token = 2 * pre_token * rec_token / (pre_token + rec_token + 1e-6)

        pre_entity = tp_entity / (tp_entity + fp_entity + 1e-6)
        rec_entity = tp_entity / (tp_entity + fn_entity + 1e-6)
        f1_entity = 2 * pre_entity * rec_entity / (
            pre_entity + rec_entity + 1e-6)

        return f1_token, f1_entity

    def _get_cascade_f1(self, preds, labels, mask):

        metrics = {}
        for i, entity_type in enumerate(self.entity_types):
            B_id = 1 + i * 4
            f1_token, f1_entity = self._get_f1(
                preds, labels, mask, B_id=B_id)
            metrics['f1 (%s-T)' % entity_type] = f1_token
            metrics['f1 (%s-E)' % entity_type] = f1_entity
        return metrics

    def _get_entities(self, ids, B_id=None):
        if not B_id:
            B_id = self.B_ID
            I_id = self.I_ID
            E_id = self.E_ID
            S_id = self.S_ID
        else:
            I_id = B_id + 1
            E_id = B_id + 2
            S_id = B_id + 3

        entities = []
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

    def _get_em_and_f1(self, preds, labels):
        em = 0
        tp, fp, fn = 0, 0, 0
        for _preds, _labels in zip(preds, labels):
            start_pred, end_pred = int(_preds[0]), int(_preds[1])
            start_label, end_label = int(_labels[0]), int(_labels[1])

            # no answer prediction
            if start_pred == 0 and end_pred == 0:
                if start_label == 0:
                    em += 1
                continue

            # invalid prediction
            if start_pred == 0 or end_pred == 0 or start_pred > end_pred:
                if start_label > 0:
                    fn += (end_label + 1 - start_label)
                continue

            # answer prediction
            tp += max(0, end_pred + 1 - start_label)
            _fp = (max(0, end_pred - end_label) +
                   max(0, start_label - start_pred))
            _fn = (max(0, start_pred - start_label) +
                   max(0, end_label - end_pred))
            if _fp + _fn == 0:
                em += 1
            fp += _fp
            fn += _fn

        exact_match = em / len(labels)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return exact_match, f1


class MTModule(BaseModule):

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

    def score(self, *args, **kwargs):
        raise AttributeError(
            '`score` method is not supported for unsupervised '
            'language modeling (LM) modules.')
