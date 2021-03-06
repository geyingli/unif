# coding:=utf-8
# Copyright 2021 Tencent. All rights reserved.
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
''' UniLM, a unified model of bidirectional modeling, unidirectional modeling
and sequence-to-sequence modeling. '''

from ..tools import tf
from .base import BaseEncoder
from .bert import BERTEncoder


class UniLMEncoder(BERTEncoder, BaseEncoder):
    def __init__(self,
                 mode,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 scope='bert',
                 drop_pooler=False,
                 trainable=True,
                 **kwargs):
        self._mode = mode
        super().__init__(
            bert_config,
            is_training,
            input_ids,
            input_mask,
            segment_ids,
            scope,
            drop_pooler,
            trainable,
            **kwargs)

    def create_attention_mask_from_input_mask(self,
                                              input_mask,
                                              batch_size,
                                              max_seq_length,
                                              dtype=tf.float32):
        if self._mode == 'bi':
            to_mask = tf.cast(tf.reshape(
                input_mask, [batch_size, 1, max_seq_length]), dtype=dtype)
            broadcast_ones = tf.ones(
                shape=[batch_size, max_seq_length, 1], dtype=dtype)
            mask = broadcast_ones * to_mask

        elif self._mode == 'l2r':
            arange = tf.range(max_seq_length) + 1
            to_mask = tf.cast(tf.sequence_mask(arange, max_seq_length), dtype)
            to_mask = tf.reshape(to_mask, [1, max_seq_length, max_seq_length])
            mask = tf.tile(to_mask, [batch_size, 1, 1])

        elif self._mode == 'r2l':
            to_mask = tf.cast(tf.reshape(
                input_mask, [batch_size, 1, max_seq_length]), dtype=dtype)
            broadcast_ones = tf.ones(
                shape=[batch_size, max_seq_length, 1], dtype=dtype)
            cover_mask = broadcast_ones * to_mask

            arange = tf.range(max_seq_length)
            reverse = tf.cast(tf.sequence_mask(arange, max_seq_length), dtype)
            reverse = tf.reshape(reverse, [1, max_seq_length, max_seq_length])
            reverse_mask = tf.tile(reverse, [batch_size, 1, 1])

            mask = (1 - reverse_mask) * cover_mask

        elif self._mode == 's2s':
            mask = tf.cast(
                tf.sequence_mask(input_mask, max_seq_length), dtype)

        return mask
