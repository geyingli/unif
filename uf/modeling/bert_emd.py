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

            # defind emd weight
            M = bert_config.num_hidden_layers
            N = student_config.num_hidden_layers
            teacher_weight = tf.get_variable(
                'teacher_weight', shape=[M],
                initializer=tf.constant_initializer(1 / M),
                trainable=False)
            student_weight = tf.get_variable(
                'student_weight', shape=[N],
                initializer=tf.constant_initializer(1 / N),
                trainable=False)

            # attention emd
            (attention_emd, attention_teacher_cost, attention_student_cost) = \
                self._get_attention_emd(
                    teacher, student, bert_config, student_config,
                    teacher_weight, student_weight, weights)

            # hidden emd
            (hidden_emd, hidden_teacher_cost, hidden_student_cost) = \
                self._get_hidden_emd(
                    teacher, student, bert_config, student_config,
                    teacher_weight, student_weight, weights)

            # update weights
            new_attention_teacher_weight = (
                tf.reduce_sum(attention_teacher_cost) /
                (attention_teacher_cost + 1e-6))
            new_hidden_teacher_weight = (
                tf.reduce_sum(hidden_teacher_cost) /
                (hidden_teacher_cost + 1e-6))
            new_teacher_weight = (
                0.5 * tf.nn.softmax(
                    new_attention_teacher_weight / emd_temporature) +
                0.5 * tf.nn.softmax(
                    new_hidden_teacher_weight / emd_temporature))
            new_attention_student_weight = (
                tf.reduce_sum(attention_student_cost) /
                (attention_student_cost + 1e-6))
            new_hidden_student_weight = (
                tf.reduce_sum(hidden_student_cost) /
                (hidden_student_cost + 1e-6))
            new_student_weight = (
                0.5 * tf.nn.softmax(
                    new_attention_student_weight / emd_temporature) +
                0.5 * tf.nn.softmax(
                    new_hidden_student_weight / emd_temporature))
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
                distill_loss = (
                    beta * (embedding_loss + attention_emd + hidden_emd) +
                    pred_loss)
            self.total_loss = distill_loss
            self.losses['losses'] = distill_loss

        else:
            student_probs = tf.nn.softmax(
                student_logits, axis=-1, name='probs')
            self.probs['probs'] = student_probs
            self.preds['preds'] = tf.argmax(student_probs, axis=-1)

    def _get_attention_emd(self, teacher, student,
                           bert_config, student_config,
                           teacher_weight, student_weight, weights):
        teacher_attention_scores = teacher.get_attention_scores()
        student_attention_scores = student.get_attention_scores()
        M = bert_config.num_hidden_layers
        N = student_config.num_hidden_layers

        with tf.variable_scope('attention_emd'):
            flow = tf.get_variable(
                'flow', shape=[M, N],
                initializer=tf.constant_initializer(1 / M / N))

            # Constraint 1:
            # Mapping should be positive.
            # f_{ij} >= 0
            flow = tf.cast(tf.clip_by_value(flow, 0, 1), tf.float32)

            # Constraint 2:
            # Amount of information flow can't exceed the weight of teacher.
            # \sum_j f_{ij} <= w_i^teacher
            ceil = teacher_weight
            weight = tf.reduce_sum(flow, axis=1)
            nominator = tf.expand_dims(ceil, axis=1)
#            nominator = tf.expand_dims(
#                tf.where(weight < ceil, weight, ceil), axis=1)
            denominator = tf.expand_dims(weight, axis=1)
            flow *= nominator / (denominator + 1e-6)

            # Constraint 3:
            # Amount of information flow can't exceed the weight of student.
            # \sum_i f_{ij} <= w_j^student
            ceil = student_weight
            weight = tf.reduce_sum(flow, axis=0)
            nominator = tf.expand_dims(ceil, axis=0)
#            nominator = tf.expand_dims(
#                tf.where(weight < ceil, weight, ceil), axis=0)
            denominator = tf.expand_dims(weight, axis=0)
            flow *= nominator / (denominator + 1e-6)

            # Constraint 4:
            # Amount of information flow can't exceed the weights.
            # \sum_{ij} f_{ij} <= min(\sum_i w_i^teacher, \sum_j w_j^student)
            sum_teacher_weight = tf.reduce_sum(teacher_weight)
            sum_student_weight = tf.reduce_sum(student_weight)
            nominator = tf.cond(
                tf.less(sum_teacher_weight, sum_student_weight),
                lambda:sum_teacher_weight,
                lambda:sum_student_weight)
            denominator = tf.reduce_sum(flow)
            flow *= nominator / (denominator + 1e-6)

            # MSE
            rows = []
            for m in range(M):
                cols = []
                for n in range(N):
                    mse = tf.losses.mean_squared_error(
                        teacher_attention_scores[m],
                        student_attention_scores[n],
                        weights=tf.reshape(weights, [-1, 1, 1, 1]))
                    col = tf.reshape(mse, [1, 1])
                    cols.append(col)
                row = tf.concat(cols, axis=1)
                rows.append(row)
            distance = tf.concat(rows, axis=0)

            # cost attention mechanism
            attention_teacher_cost = (tf.reduce_sum(flow, axis=1) *
                                      tf.reduce_sum(distance, axis=1) /
                                      (teacher_weight + 1e-6))
            attention_student_cost = (tf.reduce_sum(flow, axis=0) *
                                      tf.reduce_sum(distance, axis=0) /
                                      (student_weight + 1e-6))

        attention_emd = tf.reduce_sum(flow * distance) / (
            tf.reduce_sum(flow) + 1e-6)
        return attention_emd, attention_teacher_cost, attention_student_cost

    def _get_hidden_emd(self, teacher, student,
                        bert_config, student_config,
                        teacher_weight, student_weight, weights):
        teacher_hidden_layers = teacher.all_encoder_layers
        student_hidden_layers = student.all_encoder_layers
        M = bert_config.num_hidden_layers
        N = student_config.num_hidden_layers

        with tf.variable_scope('hidden_emd'):
            flow = tf.get_variable(
                'flow', shape=[M, N],
                initializer=tf.random_uniform_initializer(0, (1 / M / N)))

            # Constraint 1:
            # Mapping should be positive.
            # f_{ij} >= 0
            flow = tf.cast(tf.clip_by_value(flow, 0, 1), tf.float32)

            # Constraint 2:
            # Amount of information flow can't exceed the weight of teacher.
            # \sum_j f_{ij} <= w_i^teacher
            ceil = teacher_weight
            weight = tf.reduce_sum(flow, axis=1)
            nominator = tf.expand_dims(ceil, axis=1)
#            nominator = tf.expand_dims(
#                tf.where(weight < ceil, weight, ceil), axis=1)
            denominator = tf.expand_dims(weight, axis=1)
            flow *= nominator / (denominator + 1e-6)

            # Constraint 3:
            # Amount of information flow can't exceed the weight of student.
            # \sum_i f_{ij} <= w_j^student
            ceil = student_weight
            weight = tf.reduce_sum(flow, axis=0)
            nominator = tf.expand_dims(ceil, axis=0)
#            nominator = tf.expand_dims(
#                tf.where(weight < ceil, weight, ceil), axis=0)
            denominator = tf.expand_dims(weight, axis=0)
            flow *= nominator / (denominator + 1e-6)

            # Constraint 4:
            # Amount of information flow can't exceed the weights.
            # \sum_{ij} f_{ij} <= min(\sum_i w_i^teacher, \sum_j w_j^student)
            sum_teacher_weight = tf.reduce_sum(teacher_weight)
            sum_student_weight = tf.reduce_sum(student_weight)
            nominator = tf.cond(
                tf.less(sum_teacher_weight, sum_student_weight),
                lambda:sum_teacher_weight,
                lambda:sum_student_weight)
            denominator = tf.reduce_sum(flow)
            flow *= nominator / (denominator + 1e-6)

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
                        weights=tf.reshape(weights, [-1, 1, 1]))
                    col = tf.reshape(mse, [1, 1])
                    cols.append(col)
                row = tf.concat(cols, axis=1)
                rows.append(row)
            distance = tf.concat(rows, axis=0)

            # cost attention mechanism
            hidden_teacher_cost = (tf.reduce_sum(flow, axis=1) *
                                   tf.reduce_sum(distance, axis=1) /
                                   (teacher_weight + 1e-6))
            hidden_student_cost = (tf.reduce_sum(flow, axis=0) *
                                   tf.reduce_sum(distance, axis=0) /
                                   (student_weight + 1e-6))

        hidden_emd = tf.reduce_sum(flow * distance) / (
            tf.reduce_sum(flow) + 1e-6)
        return hidden_emd, hidden_teacher_cost, hidden_student_cost

    def _get_pred_loss(self, teacher_logits, student_logits, weights,
                       pred_temporature):
        teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
        student_log_probs = \
            tf.nn.log_softmax(student_logits, axis=-1) / pred_temporature
        pred_loss = (
            - tf.reduce_sum(teacher_probs * student_log_probs, axis=-1) *
            tf.reshape(weights, [-1, 1]))
        pred_loss = tf.reduce_mean(pred_loss)
        return pred_loss
