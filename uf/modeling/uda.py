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
''' Unsupervised Data Augmentation for Consistency Training (UDA).
  Code revised from Google's implementation.
  See `https://github.com/google-research/uda`.
'''

from uf.tools import tf
from .base import BaseDecoder
from . import util



class UDADecoder(BaseDecoder):
    def __init__(self,
                 is_training,
                 input_tensor,
                 is_supervised,
                 is_expanded,
                 label_ids,
                 label_size=2,
                 sample_weight=None,
                 scope='cls/seq_relationship',
                 hidden_dropout_prob=0.1,
                 initializer_range=0.02,
                 trainable=True,
                 global_step=None,
                 num_train_steps=None,
                 uda_softmax_temp=-1,
                 uda_confidence_thresh=-1,
                 tsa_schedule='linear',
                 **kwargs):
        super().__init__(**kwargs)

        is_supervised = tf.cast(is_supervised, tf.float32)
        is_expanded = tf.cast(is_expanded, tf.float32)

        hidden_size = input_tensor.shape.as_list()[-1]
        with tf.variable_scope(scope):
            output_weights = tf.get_variable(
                'output_weights',
                shape=[label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable)
            output_bias = tf.get_variable(
                'output_bias',
                shape=[label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable)

            output_layer = util.dropout(
                input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            with tf.variable_scope('sup_loss'):

                # reshape
                sup_ori_log_probs = tf.boolean_mask(
                    log_probs, mask=(1.0 - is_expanded), axis=0)
                sup_log_probs = tf.boolean_mask(
                    sup_ori_log_probs, mask=is_supervised, axis=0)
                sup_label_ids = tf.boolean_mask(
                    label_ids, mask=is_supervised, axis=0)

                self.preds['preds'] = tf.argmax(sup_ori_log_probs, axis=-1)

                one_hot_labels = tf.one_hot(
                    sup_label_ids, depth=label_size, dtype=tf.float32)
                per_example_loss = - tf.reduce_sum(
                    one_hot_labels * sup_log_probs, axis=-1)

                loss_mask = tf.ones_like(per_example_loss, dtype=tf.float32)
                correct_label_probs = tf.reduce_sum(
                    one_hot_labels * tf.exp(sup_log_probs), axis=-1)

                if is_training and tsa_schedule:
                    tsa_start = 1.0 / label_size
                    tsa_threshold = get_tsa_threshold(
                        tsa_schedule, global_step, num_train_steps,
                        tsa_start, end=1)

                    larger_than_threshold = tf.greater(
                        correct_label_probs, tsa_threshold)
                    loss_mask = loss_mask * (
                        1 - tf.cast(larger_than_threshold, tf.float32))

                loss_mask = tf.stop_gradient(loss_mask)
                per_example_loss = per_example_loss * loss_mask
                if sample_weight is not None:
                    sup_sample_weight = tf.boolean_mask(
                        sample_weight, mask=is_supervised, axis=0)
                    per_example_loss *= tf.cast(
                        sup_sample_weight, dtype=tf.float32)
                sup_loss = (tf.reduce_sum(per_example_loss) /
                            tf.maximum(tf.reduce_sum(loss_mask), 1))

                self.losses['supervised'] = per_example_loss

            with tf.variable_scope('unsup_loss'):

                # reshape
                ori_log_probs = tf.boolean_mask(
                    sup_ori_log_probs, mask=(1.0 - is_supervised), axis=0)
                aug_log_probs = tf.boolean_mask(
                    log_probs, mask=is_expanded, axis=0)
                sup_ori_logits = tf.boolean_mask(
                    logits, mask=(1.0 - is_expanded), axis=0)
                ori_logits = tf.boolean_mask(
                    sup_ori_logits, mask=(1.0 - is_supervised), axis=0)

                unsup_loss_mask = 1
                if uda_softmax_temp != -1:
                    tgt_ori_log_probs = tf.nn.log_softmax(
                        ori_logits / uda_softmax_temp, axis=-1)
                    tgt_ori_log_probs = tf.stop_gradient(tgt_ori_log_probs)
                else:
                    tgt_ori_log_probs = tf.stop_gradient(ori_log_probs)

                if uda_confidence_thresh != -1:
                    largest_prob = tf.reduce_max(tf.exp(ori_log_probs), axis=-1)
                    unsup_loss_mask = tf.cast(tf.greater(
                        largest_prob, uda_confidence_thresh), tf.float32)
                    unsup_loss_mask = tf.stop_gradient(unsup_loss_mask)

                per_example_loss = kl_for_log_probs(
                    tgt_ori_log_probs, aug_log_probs) * unsup_loss_mask
                if sample_weight is not None:
                    unsup_sample_weight = tf.boolean_mask(
                        sample_weight, mask=(1.0 - is_supervised), axis=0)
                    per_example_loss *= tf.cast(
                        unsup_sample_weight, dtype=tf.float32)
                unsup_loss = tf.reduce_mean(per_example_loss)

                self.losses['unsupervised'] = per_example_loss

            self.total_loss = sup_loss + unsup_loss



def get_tsa_threshold(tsa_schedule, global_step, num_train_steps, start, end):
    training_progress = tf.to_float(global_step) / tf.to_float(num_train_steps)
    if tsa_schedule == 'linear':
        threshold = training_progress
    elif tsa_schedule == 'exp':
        scale = 5
        threshold = tf.exp((training_progress - 1) * scale)
        # [exp(-5), exp(0)] = [1e-2, 1]
    elif tsa_schedule == 'log':
        scale = 5
        # [1 - exp(0), 1 - exp(-5)] = [0, 0.99]
        threshold = 1 - tf.exp((-training_progress) * scale)
    else:
        raise ValueError(
            'Invalid value for `tsa_schedule`: %s. Pick one from `linear`, '
            '`exp` or `log`.' % (tsa_schedule))
    return threshold * (end - start) + start


def kl_for_log_probs(log_p, log_q):
    p = tf.exp(log_p)
    neg_ent = tf.reduce_sum(p * log_p, axis=-1)
    neg_cross_ent = tf.reduce_sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl
