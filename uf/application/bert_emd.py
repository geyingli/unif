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
''' Applications based on BERT-EMD. '''

import copy

from .base import ClassifierModule
from uf.modeling.bert_emd import BERTEMDCLSDistillor
from .bert import get_bert_config
from .tiny_bert import TinyBERTClassifier
from uf.tokenization.word_piece import get_word_piece_tokenizer
from pyemd import emd_with_flow



class BERTEMDClassifier(TinyBERTClassifier, ClassifierModule):
    ''' Single-label classifier on BERT-EMD, an advanced distillation model
    of TinyBERT. '''
    _INFER_ATTRIBUTES = TinyBERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 hidden_size=384,
                 num_hidden_layers=4,
                 pred_temporature=1.0,
                 emd_temporature=1.0,
                 beta=0.01,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self.pred_temporature = pred_temporature
        self.emd_temporature = emd_temporature
        self.beta = beta
        self._drop_pooler = drop_pooler
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = 'unsupported'

        self.student_config = copy.deepcopy(self.bert_config)
        self.student_config.hidden_size = hidden_size
        self.student_config.intermediate_size = 4 * hidden_size
        self.student_config.num_hidden_layers = num_hidden_layers

        if '[CLS]' not in self.tokenizer.vocab:
            self.tokenizer.add('[CLS]')
            self.bert_config.vocab_size += 1
        if '[SEP]' not in self.tokenizer.vocab:
            self.tokenizer.add('[SEP]')
            self.bert_config.vocab_size += 1

    def to_bert(self):
        ''' Isolate student bert_emd out of traing graph. '''
        super().to_bert()

    def _forward(self, is_training, split_placeholders, **kwargs):

        distillor = BERTEMDCLSDistillor(
            student_config=self.student_config,
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='bert',
            drop_pooler=self._drop_pooler,
            label_size=self.label_size,
            pred_temporature=self.pred_temporature,
            emd_temporature=self.emd_temporature,
            beta=self.beta,
            **kwargs)
        if is_training:
            self._emd_tensors = [emd_with_flow] + distillor.get_emd_tensors()
        (total_loss, losses, probs, preds) = distillor.get_forward_outputs()
        return (total_loss, losses, probs, preds)
