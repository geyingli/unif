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
''' BERT-EMD, an advanced distillation model of TinyBERT. '''

from uf.tools import tf
from .bert import BERTEncoder
from .tiny_bert import TinyBERTCLSDistillor
from . import util


class BERTEMDCLSDistillor(TinyBERTCLSDistillor):
    def __init__(self,
                 student_config,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids=None,
                 sample_weight=None,
                 scope='bert',
                 dtype=tf.float32,
                 drop_pooler=False,
                 label_size=2,
                 pred_temporature=1.0,
                 emd_temporature=1.0,
                 beta=0.01,
                 **kwargs):
        super(TinyBERTCLSDistillor, self).__init__()

        def _get_logits(pooled_output, hidden_size, scope, trainable):
            with tf.variable_scope(scope):
                output_weights = tf.get_variable(
                    'output_weights',
                    shape=[label_size, hidden_size],
                    initializer=util.create_initializer(
                        bert_config.initializer_range),
                    trainable=trainable)
                output_bias = tf.get_variable(
                    'output_bias',
                    shape=[label_size],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable)

                logits = tf.matmul(pooled_output,
                                   output_weights,
                                   transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                return logits

        use_tilda_embedding=kwargs.get('use_tilda_embedding')
        student = BERTEncoder(
            bert_config=student_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            scope='tiny/bert',
            use_tilda_embedding=use_tilda_embedding,
            drop_pooler=drop_pooler,
            trainable=True,
            **kwargs)
        student_logits = _get_logits(
            student.get_pooled_output(),
            student_config.hidden_size, 'tiny/cls/seq_relationship', True)

        if is_training:
            teacher = BERTEncoder(
                bert_config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                scope=scope,
                use_tilda_embedding=False,
                drop_pooler=drop_pooler,
                trainable=False,
                **kwargs)
            teacher_logits = _get_logits(
                teacher.get_pooled_output(),
                bert_config.hidden_size, 'cls/seq_relationship', False)

            weights = 1.0
            if sample_weight is not None:
                weights = tf.cast(sample_weight, dtype=tf.float32)

            # embedding loss
            embedding_loss = self._get_embedding_loss(
                teacher, student, bert_config, weights)

            # emd
            M = bert_config.num_hidden_layers
            N = student_config.num_hidden_layers
            with tf.variable_scope('emd'):
                teacher_weight = tf.get_variable(
                    'teacher_weight', shape=[M],
                    initializer=tf.constant_initializer(1 / M),
                    trainable=False)
                student_weight = tf.get_variable(
                    'student_weight', shape=[N],
                    initializer=tf.constant_initializer(1 / N),
                    trainable=False)
                self.teacher_weight = teacher_weight
                self.student_weight = student_weight

            # attention emd
            (attention_emd, new_attention_teacher_weight,
             new_attention_student_weight) = \
                self._get_attention_emd(
                    teacher, student, teacher_weight, student_weight,
                    weights, emd_temporature)

            # hidden emd
            (hidden_emd, new_hidden_teacher_weight,
             new_hidden_student_weight) = \
                self._get_hidden_emd(
                    teacher, student, teacher_weight, student_weight,
                    bert_config, weights, emd_temporature)

            # update weights
            new_teacher_weight = \
                (new_attention_teacher_weight + new_hidden_teacher_weight) / 2
            new_student_weight = \
                (new_attention_student_weight + new_hidden_student_weight) / 2
            update_teacher_weight_op = tf.assign(
                teacher_weight, new_teacher_weight)
            update_student_weight_op = tf.assign(
                student_weight, new_student_weight)

            # prediction loss
            pred_loss = self._get_pred_loss(
                teacher_logits, student_logits, weights, pred_temporature)

            # sum up
            with tf.control_dependencies([update_teacher_weight_op,
                                          update_student_weight_op]):
                distill_loss = \
                    beta * (embedding_loss + attention_emd + hidden_emd) + \
                    pred_loss
            self.total_loss = distill_loss
            self.losses['losses'] = distill_loss

        else:
            student_probs = tf.nn.softmax(
                student_logits, axis=-1, name='probs')
            self.probs['probs'] = student_probs
            self.preds['preds'] = tf.argmax(student_probs, axis=-1)

    def _get_attention_emd(self, teacher, student,
                           teacher_weight, student_weight,
                           sample_weight, emd_temporature):
        teacher_attention_scores = teacher.get_attention_scores()
        teacher_attention_scores = [tf.stop_gradient(value)
                                    for value in teacher_attention_scores]
        student_attention_scores = student.get_attention_scores()
        M = len(teacher_attention_scores)
        N = len(student_attention_scores)

        with tf.variable_scope('attention_emd'):
            flow = tf.get_variable(
                'flow', shape=[M, N],
                initializer=tf.constant_initializer(1 / M / N),
                trainable=False)

            # MSE
            rows = []
            for m in range(M):
                cols = []
                for n in range(N):
                    mse = tf.losses.mean_squared_error(
                        teacher_attention_scores[m],
                        student_attention_scores[n],
                        weights=tf.reshape(sample_weight, [-1, 1, 1, 1]))
                    col = tf.reshape(mse, [1, 1])
                    cols.append(col)
                row = tf.concat(cols, axis=1)
                rows.append(row)
            distance = tf.concat(rows, axis=0)

            # cost attention mechanism
            teacher_cost = (tf.reduce_sum(flow, axis=1) *
                            tf.reduce_sum(distance, axis=1) /
                            (teacher_weight + 1e-6))
            student_cost = (tf.reduce_sum(flow, axis=0) *
                            tf.reduce_sum(distance, axis=0) /
                            (student_weight + 1e-6))

            # new weights
            new_teacher_weight = tf.where(
                teacher_cost > 1e-12,
                tf.reduce_sum(teacher_cost) / (teacher_cost + 1e-6),
                teacher_weight)
            new_student_weight = tf.where(
                student_cost > 1e-12,
                tf.reduce_sum(student_cost) / (student_cost + 1e-6),
                student_weight)
            new_teacher_weight = tf.nn.softmax(
                new_teacher_weight / emd_temporature)
            new_student_weight = tf.nn.softmax(
                new_student_weight / emd_temporature)

        self.attention_flow = flow
        self.attention_distance = distance
        attention_emd = tf.reduce_sum(flow * distance)
        return attention_emd, new_teacher_weight, new_student_weight

    def _get_hidden_emd(self, teacher, student,
                        teacher_weight, student_weight,
                        bert_config, sample_weight, emd_temporature):
        teacher_hidden_layers = teacher.all_encoder_layers
        teacher_hidden_layers = [tf.stop_gradient(value)
                                 for value in teacher_hidden_layers]
        student_hidden_layers = student.all_encoder_layers
        M = len(teacher_hidden_layers)
        N = len(student_hidden_layers)

        with tf.variable_scope('hidden_emd'):
            flow = tf.get_variable(
                'flow', shape=[M, N],
                initializer=tf.constant_initializer(1 / M / N),
                trainable=False)

            # MSE
            rows = []
            for m in range(M):
                cols = []
                for n in range(N):
                    linear_trans = tf.layers.dense(
                        student_hidden_layers[n],
                        bert_config.hidden_size,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range))
                    mse = tf.losses.mean_squared_error(
                        teacher_hidden_layers[m], linear_trans,
                        weights=tf.reshape(sample_weight, [-1, 1, 1]))
                    col = tf.reshape(mse, [1, 1])
                    cols.append(col)
                row = tf.concat(cols, axis=1)
                rows.append(row)
            distance = tf.concat(rows, axis=0)

            # cost attention mechanism
            teacher_cost = (tf.reduce_sum(flow, axis=1) *
                            tf.reduce_sum(distance, axis=1) /
                            (teacher_weight + 1e-6))
            student_cost = (tf.reduce_sum(flow, axis=0) *
                            tf.reduce_sum(distance, axis=0) /
                            (student_weight + 1e-6))

            # new weights
            new_teacher_weight = tf.where(
                teacher_cost > 1e-12,
                tf.reduce_sum(teacher_cost) / (teacher_cost + 1e-6),
                teacher_weight)
            new_student_weight = tf.where(
                student_cost > 1e-12,
                tf.reduce_sum(student_cost) / (student_cost + 1e-6),
                student_weight)
            new_teacher_weight = tf.nn.softmax(
                new_teacher_weight / emd_temporature)
            new_student_weight = tf.nn.softmax(
                new_student_weight / emd_temporature)

        self.hidden_flow = flow
        self.hidden_distance = distance
        hidden_emd = tf.reduce_sum(flow * distance)
        return hidden_emd, new_teacher_weight, new_student_weight

    def _get_pred_loss(self, teacher_logits, student_logits, weights,
                       pred_temporature):
        teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
        teacher_probs = tf.stop_gradient(teacher_probs)
        student_log_probs = \
            tf.nn.log_softmax(student_logits, axis=-1) / pred_temporature
        pred_loss = (
            - tf.reduce_sum(teacher_probs * student_log_probs, axis=-1) *
            tf.reshape(weights, [-1, 1]))
        pred_loss = tf.reduce_mean(pred_loss)
        return pred_loss

    def get_emd_tensors(self):
        return [self.attention_flow,
                self.attention_distance,
                self.hidden_flow,
                self.hidden_distance,
                self.teacher_weight,
                self.student_weight,]

