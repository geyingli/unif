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
''' Applications based on Performer. '''

from .base import ClassifierModule
from .bert import BERTClassifier
from uf.modeling.performer import PerformerEncoder
from uf.modeling.base import CLSDecoder



class PerformerClassifier(BERTClassifier, ClassifierModule):
    ''' Single-label classifier on Performer. '''
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 kernel_transformation='relu',
                 nb_random_features=1,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super().__init__(
            config_file, vocab_file,
            max_seq_length=max_seq_length,
            label_size=label_size,
            init_checkpoint=init_checkpoint,
            output_dir=output_dir,
            gpu_ids=gpu_ids,
            drop_pooler=drop_pooler,
            do_lower_case=do_lower_case,
            truncate_method=truncate_method)
        self._kernel_transformation = kernel_transformation
        self._nb_random_features = nb_random_features
        self._id_to_label = None
        self.__init_args__ = locals()

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = PerformerEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            segment_ids=split_placeholders['segment_ids'],
            scope='performer',
            kernel_transformation=self._kernel_transformation,
            nb_random_features=self._nb_random_features,
            drop_pooler=self._drop_pooler,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = CLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/seq_relationship',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)
