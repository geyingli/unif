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
''' TinyBERT, a self-distillation model combining BERT with dynamic inference.
'''

from uf.tools import tf
from .bert import BERTEncoder
from .base import BaseDecoder
from . import util


class TinyBERTCLSDistillor(BaseDecoder):
    def __init__(self,
                 tiny_bert_config,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids=None,
                 sample_weight=None,
                 scope='bert',
                 name='',
                 dtype=tf.float32,
                 use_tilda_embedding=False,
                 drop_pooler=False,
                 label_size=2,
                 trainable=True,
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

        student = BERTEncoder(
            bert_config=tiny_bert_config,
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
            tiny_bert_config.hidden_size, 'tiny/cls/seq_relationship', True)

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
            teacher_embedding = teacher.get_embedding_output()
            student_embedding = student.get_embedding_output()
            with tf.variable_scope('embedding_loss'):
                linear_trans = tf.layers.dense(
                    student_embedding,
                    bert_config.hidden_size,
                    kernel_initializer=util.create_initializer(
                        bert_config.initializer_range))
                embedding_loss = tf.losses.mean_squared_error(
                    linear_trans, teacher_embedding,
                    weights=tf.reshape(weights, [-1, 1, 1]))

            # attention loss
            teacher_attention_scores = teacher.get_attention_scores()
            student_attention_scores = student.get_attention_scores()
            num_teacher_hidden_layers = bert_config.num_hidden_layers
            num_student_hidden_layers = tiny_bert_config.num_hidden_layers
            num_projections = \
                int(num_teacher_hidden_layers / num_student_hidden_layers)
            attention_losses = []
            for i in range(num_student_hidden_layers):
                attention_losses.append(tf.losses.mean_squared_error(
                    teacher_attention_scores[
                        num_projections * i + num_projections - 1],
                    student_attention_scores[i],
                    weights=tf.reshape(weights, [-1, 1, 1, 1])),)
            attention_loss = tf.add_n(attention_losses)

            # hidden loss
            teacher_hidden_layers = teacher.all_encoder_layers
            student_hidden_layers = student.all_encoder_layers
            num_teacher_hidden_layers = bert_config.num_hidden_layers
            num_student_hidden_layers = tiny_bert_config.num_hidden_layers
            num_projections = int(
                num_teacher_hidden_layers / num_student_hidden_layers)
            with tf.variable_scope('hidden_loss'):
                hidden_losses = []
                for i in range(num_student_hidden_layers):
                    hidden_losses.append(tf.losses.mean_squared_error(
                        teacher_hidden_layers[
                            num_projections * i + num_projections - 1],
                        tf.layers.dense(
                            student_hidden_layers[i], bert_config.hidden_size,
                            kernel_initializer=util.create_initializer(
                                bert_config.initializer_range)),
                        weights=tf.reshape(weights, [-1, 1, 1])))
                hidden_loss = tf.add_n(hidden_losses)

            # prediction loss
            teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
            student_log_probs = tf.nn.log_softmax(student_logits, axis=-1)
            pred_loss = (
                - tf.reduce_sum(teacher_probs * student_log_probs, axis=-1) *
                tf.reshape(weights, [-1, 1]))
            pred_loss = tf.reduce_mean(pred_loss)

            # sum up
            distill_loss = (embedding_loss + attention_loss + hidden_loss +
                            pred_loss)
            self.total_loss = distill_loss
            self.losses['distill'] = tf.reshape(distill_loss, [1])

        else:
            student_probs = tf.nn.softmax(
                student_logits, axis=-1, name='probs')
            self.probs[name] = student_probs



class TinyBERTREGDistillor(BaseDecoder):
    pass