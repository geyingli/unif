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
''' FastBERT, self-distilled BERT with dynamic inference.
  Code revised from Google's implementation of BERT.
  See `https://github.com/google-research/bert`.
'''

import copy
import collections

from uf.tools import tf
from .bert import BERTEncoder
from .base import BaseDecoder
from . import util


class FastBERTCLSDistillor(BaseDecoder, BERTEncoder):
    def __init__(self,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 sample_weight=None,
                 scope='bert',
                 name='',
                 dtype=tf.float32,
                 drop_pooler=False,
                 cls_model='self-attention',
                 label_size=2,
                 speed=0.1,
                 ignore_cls=None,
                 trainable=True,
                 **kwargs):
        super(FastBERTCLSDistillor, self).__init__()

        bert_config = copy.deepcopy(bert_config)
        bert_config.hidden_dropout_prob = 0.0
        bert_config.attention_probs_dropout_prob = 0.0

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        with tf.variable_scope(scope):
            with tf.variable_scope('embeddings'):

                (self.embedding_output, self.embedding_table) = \
                    self.embedding_lookup(
                        input_ids=input_ids,
                        vocab_size=bert_config.vocab_size,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length,
                        embedding_size=bert_config.hidden_size,
                        initializer_range=bert_config.initializer_range,
                        word_embedding_name='word_embeddings',
                        dtype=dtype,
                        trainable=False,
                        tilda_embeddings=None)

                # Add positional embeddings and token type embeddings
                # layer normalize and perform dropout.
                self.embedding_output = self.embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    hidden_size=bert_config.hidden_size,
                    use_token_type=True,
                    segment_ids=segment_ids,
                    token_type_vocab_size=bert_config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=bert_config.initializer_range,
                    max_position_embeddings=\
                        bert_config.max_position_embeddings,
                    dropout_prob=bert_config.hidden_dropout_prob,
                    dtype=dtype,
                    trainable=False)

            with tf.variable_scope('encoder'):
                attention_mask = self.create_attention_mask_from_input_mask(
                    input_mask, batch_size, max_seq_length, dtype=dtype)

                # stacked transformers
                (self.all_encoder_layers, self.all_cls_layers) = \
                    self.dynamic_transformer_model(
                        is_training,
                        input_tensor=self.embedding_output,
                        input_mask=input_mask,
                        batch_size=batch_size,
                        max_seq_length=max_seq_length,
                        label_size=label_size,
                        attention_mask=attention_mask,
                        hidden_size=bert_config.hidden_size,
                        num_hidden_layers=bert_config.num_hidden_layers,
                        num_attention_heads=bert_config.num_attention_heads,
                        intermediate_size=bert_config.intermediate_size,
                        intermediate_act_fn=util.get_activation(
                            bert_config.hidden_act),
                        hidden_dropout_prob=bert_config.hidden_dropout_prob,
                        attention_probs_dropout_prob=\
                            bert_config.attention_probs_dropout_prob,
                        initializer_range=bert_config.initializer_range,
                        dtype=dtype,
                        trainable=False,
                        cls_model=cls_model,
                        speed=speed,
                        ignore_cls=ignore_cls)

            self.sequence_output = self.all_encoder_layers[-1]
            with tf.variable_scope('pooler'):
                first_token_tensor = self.sequence_output[:, 0, :]

                # trick: ignore the fully connected layer
                if drop_pooler:
                    self.pooled_output = first_token_tensor
                else:
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        bert_config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=util.create_initializer(
                            bert_config.initializer_range),
                        trainable=False)

        # teacher classifier
        if bert_config.num_hidden_layers not in ignore_cls:
            with tf.variable_scope('cls/seq_relationship'):
                output_weights = tf.get_variable(
                    'output_weights',
                    shape=[label_size, bert_config.hidden_size],
                    initializer=util.create_initializer(
                        bert_config.initializer_range),
                    trainable=False)
                output_bias = tf.get_variable(
                    'output_bias',
                    shape=[label_size],
                    initializer=tf.zeros_initializer(),
                    trainable=False)

                logits = tf.matmul(self.pooled_output,
                                   output_weights,
                                   transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                probs = tf.nn.softmax(logits, axis=-1)

        # distillation
        if is_training:
            losses = []
            for cls_probs in self.all_cls_layers.values():

                # KL-Divergence
                per_example_loss = tf.reduce_sum(
                    cls_probs * (tf.log(cls_probs) - tf.log(probs)), axis=-1)
                if sample_weight is not None:
                    per_example_loss = \
                        (tf.cast(sample_weight, dtype=tf.float32) *
                         per_example_loss)
                loss = tf.reduce_mean(per_example_loss)
                losses.append(loss)

            distill_loss = tf.add_n(losses)
            self.total_loss = distill_loss
            self.losses['distill'] = distill_loss

        else:
            if bert_config.num_hidden_layers not in ignore_cls:
                self.all_cls_layers[bert_config.num_hidden_layers] = probs
            self.probs[name] = tf.concat(
                list(self.all_cls_layers.values()), axis=0, name='probs')

    def dynamic_transformer_model(self,
                                  is_training,
                                  input_tensor,
                                  input_mask,
                                  batch_size,
                                  max_seq_length,
                                  label_size,
                                  attention_mask=None,
                                  hidden_size=768,
                                  num_hidden_layers=12,
                                  num_attention_heads=12,
                                  intermediate_size=3072,
                                  intermediate_act_fn=util.gelu,
                                  hidden_dropout_prob=0.1,
                                  attention_probs_dropout_prob=0.1,
                                  initializer_range=0.02,
                                  dtype=tf.float32,
                                  trainable=True,
                                  cls_model='self-attention',
                                  cls_hidden_size=128,
                                  cls_num_attention_heads=2,
                                  speed=0.1,
                                  ignore_cls=None):
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of '
                'attention heads (%d)'
                % (hidden_size, num_attention_heads))
        attention_head_size = int(hidden_size / num_attention_heads)

        keep_cls = list(range(num_hidden_layers + 1))
        keep_cls = [
            cls_idx for cls_idx in keep_cls if cls_idx not in ignore_cls]

        all_layer_outputs = []
        all_layer_cls_outputs = collections.OrderedDict()
        prev_output = input_tensor
        prev_mask = input_mask
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope('layer_%d' % layer_idx):

                # build child classifier
                if is_training or layer_idx not in ignore_cls:
                    with tf.variable_scope('distill'):

                        # FCN + Self_Attention + FCN + FCN
                        if cls_model == 'self-attention-paper':
                            cls_output = self._cls_self_attention_paper(
                                prev_output,
                                batch_size,
                                max_seq_length,
                                label_size,
                                attention_mask=None,
                                cls_hidden_size=cls_hidden_size,
                                cls_num_attention_heads=\
                                    cls_num_attention_heads,
                                attention_probs_dropout_prob=\
                                    attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                dtype=tf.float32,
                                trainable=True)

                        # Self_Attention + FCN
                        elif cls_model == 'self-attention':
                            cls_output = self._cls_self_attention(
                                prev_output,
                                batch_size,
                                max_seq_length,
                                label_size,
                                attention_mask=None,
                                cls_hidden_size=cls_hidden_size,
                                cls_num_attention_heads=\
                                    cls_num_attention_heads,
                                attention_probs_dropout_prob=\
                                    attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                dtype=tf.float32,
                                trainable=True)

                        # FCN
                        elif cls_model == 'fcn':
                            cls_output = self._cls_fcn(
                                prev_output,
                                label_size,
                                hidden_size=hidden_size,
                                initializer_range=initializer_range,
                                dtype=tf.float32,
                                trainable=True)

                        else:
                            raise ValueError(
                                'Invalid `cls_model = %s`. Pick one from '
                                '`self-attention-paper`, `self-attention` '
                                'and `fcn`' % cls_model)

                        # distill core
                        layer_cls_output = tf.nn.softmax(
                            cls_output, axis=-1, name='cls_%d' % layer_idx)
                        uncertainty = tf.reduce_sum(layer_cls_output * tf.log(
                            layer_cls_output), axis=-1)
                        uncertainty /= tf.log(1 / label_size)

                    # branching only in inference
                    if not is_training:

                        # last output
                        if layer_idx == keep_cls[-1]:
                            all_layer_outputs.append(prev_output)
                            all_layer_cls_outputs[layer_idx] = layer_cls_output
                            return (all_layer_outputs, all_layer_cls_outputs)

                        mask = tf.less(uncertainty, speed)
                        unfinished_mask = \
                            (tf.ones_like(mask, dtype=dtype) -
                             tf.cast(mask, dtype=dtype))
                        prev_output = tf.boolean_mask(
                            prev_output, mask=unfinished_mask, axis=0)
                        prev_mask = tf.boolean_mask(
                            prev_mask, mask=unfinished_mask, axis=0)
                    all_layer_cls_outputs[layer_idx] = layer_cls_output

                    # new attention mask
                    input_shape = util.get_shape_list(prev_output)
                    batch_size = input_shape[0]
                    max_seq_length = input_shape[1]
                    attention_mask = \
                        self.create_attention_mask_from_input_mask(
                            prev_mask, batch_size, max_seq_length, dtype=dtype)

                # originial stream
                with tf.variable_scope('attention'):
                    attention_heads = []
                    with tf.variable_scope('self'):
                        (attention_head, _) = self.attention_layer(
                            from_tensor=prev_output,
                            to_tensor=prev_output,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=\
                                attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=False,
                            batch_size=batch_size,
                            from_max_seq_length=max_seq_length,
                            to_max_seq_length=max_seq_length,
                            dtype=dtype,
                            trainable=False)
                        attention_heads.append(attention_head)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        attention_output = tf.concat(attention_heads, axis=-1)

                    with tf.variable_scope('output'):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=util.create_initializer(
                                initializer_range),
                            trainable=False)
                        attention_output = util.dropout(
                            attention_output, hidden_dropout_prob)
                        attention_output = util.layer_norm(
                            attention_output + prev_output, trainable=False)

                # The activation is only applied to the `intermediate`
                # hidden layer.
                with tf.variable_scope('intermediate'):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=util.create_initializer(
                            initializer_range),
                        trainable=False)

                # Down-project back to hidden_size then add the residual.
                with tf.variable_scope('output'):
                    layer_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=util.create_initializer(
                            initializer_range),
                        trainable=False)
                    layer_output = util.dropout(
                        layer_output, hidden_dropout_prob)
                    layer_output = util.layer_norm(
                        layer_output + attention_output, trainable=False)

                prev_output = layer_output
                all_layer_outputs.append(layer_output)

        return (all_layer_outputs, all_layer_cls_outputs)

    def _cls_self_attention_paper(self,
                                  prev_output,
                                  batch_size,
                                  max_seq_length,
                                  label_size,
                                  attention_mask=None,
                                  cls_hidden_size=128,
                                  cls_num_attention_heads=2,
                                  attention_probs_dropout_prob=0.1,
                                  initializer_range=0.02,
                                  dtype=tf.float32,
                                  trainable=True):
        if cls_hidden_size % cls_num_attention_heads != 0:
            raise ValueError(
                '`cls_hidden_size` (%d) is not a multiple of the number of '
                '`cls_num_attention_heads` (%d)'
                % (cls_hidden_size, cls_num_attention_heads))
        cls_attention_head_size = int(
            cls_hidden_size / cls_num_attention_heads)

        with tf.variable_scope('project'):
            attention_input = tf.layers.dense(
                prev_output,
                cls_hidden_size,
                activation='tanh',
                kernel_initializer=util.create_initializer(
                    initializer_range),
                trainable=trainable)

        with tf.variable_scope('attention'):
            attention_heads = []
            with tf.variable_scope('self'):
                (attention_head, _) = self.attention_layer(
                    from_tensor=attention_input,
                    to_tensor=attention_input,
                    attention_mask=attention_mask,
                    num_attention_heads=cls_num_attention_heads,
                    size_per_head=cls_attention_head_size,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_max_seq_length=max_seq_length,
                    to_max_seq_length=max_seq_length,
                    dtype=dtype,
                    trainable=trainable)
                attention_heads.append(attention_head)

            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)

        with tf.variable_scope('intermediate'):
            intermediate_output = tf.layers.dense(
                attention_output[:, 0, :],
                cls_hidden_size,
                activation='tanh',
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable)

        with tf.variable_scope('output'):
            cls_output = tf.layers.dense(
                intermediate_output,
                label_size,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable)

        return cls_output

    def _cls_self_attention(self,
                            prev_output,
                            batch_size,
                            max_seq_length,
                            label_size,
                            attention_mask=None,
                            cls_hidden_size=128,
                            cls_num_attention_heads=2,
                            attention_probs_dropout_prob=0.1,
                            initializer_range=0.02,
                            dtype=tf.float32,
                            trainable=True):
        if cls_hidden_size % cls_num_attention_heads != 0:
            raise ValueError(
                '`cls_hidden_size` (%d) is not a multiple of the number of '
                '`cls_num_attention_heads` (%d)'
                % (cls_hidden_size, cls_num_attention_heads))
        cls_attention_head_size = int(
            cls_hidden_size / cls_num_attention_heads)

        with tf.variable_scope('attention'):
            attention_heads = []
            with tf.variable_scope('self'):
                attention_head, _ = self.attention_layer(
                    from_tensor=prev_output,
                    to_tensor=prev_output,
                    attention_mask=attention_mask,
                    num_attention_heads=cls_num_attention_heads,
                    size_per_head=cls_attention_head_size,
                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                    initializer_range=initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_max_seq_length=max_seq_length,
                    to_max_seq_length=max_seq_length,
                    dtype=dtype,
                    trainable=trainable)
                attention_heads.append(attention_head)

            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                attention_output = tf.concat(attention_heads, axis=-1)
            attention_output = util.layer_norm(
                attention_output[:, 0, :], trainable=trainable)

        with tf.variable_scope('output'):
            cls_output = tf.layers.dense(
                attention_output,
                label_size,
                kernel_initializer=util.create_initializer(initializer_range),
                trainable=trainable)

        return cls_output

    def _cls_fcn(self,
                 prev_output,
                 label_size,
                 hidden_size=768,
                 initializer_range=0.02,
                 dtype=tf.float32,
                 trainable=True):

        with tf.variable_scope('output'):
            cls_output_weights = tf.get_variable(
                'output_weights', [hidden_size, label_size],
                initializer=tf.truncated_normal_initializer(
                    stddev=initializer_range),
                dtype=dtype,
                trainable=trainable)
            cls_output_bias = tf.get_variable(
                'output_bias', [label_size],
                initializer=tf.zeros_initializer(),
                dtype=dtype,
                trainable=trainable)
            cls_logits = tf.matmul(prev_output[:, 0, :], cls_output_weights)
            cls_output = tf.nn.bias_add(cls_logits, cls_output_bias)

        return cls_output
