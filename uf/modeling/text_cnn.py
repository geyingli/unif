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
''' Convolutional neural network on texture analysis. '''

from uf.tools import tf
from .base import BaseEncoder
from . import util


class TextCNNEncoder(BaseEncoder):
    def __init__(self,
                 vocab_size,
                 filter_sizes,
                 num_channels,
                 is_training,
                 input_ids,
                 scope='text_cnn',
                 embedding_size=256,
                 dropout_prob=0.1,
                 trainable=True,
                 **kwargs):

        input_shape = util.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        max_seq_length = input_shape[1]

        if isinstance(filter_sizes, str):
            filter_sizes = filter_sizes.split(',')
        assert isinstance(filter_sizes, list), (
            '`filter_sizes` should be a list of integers or a string '
            'seperated with commas.')

        # Tilda embeddings for SMART algorithm
        tilda_embeddings = None
        use_tilda_embedding=kwargs.get('use_tilda_embedding')
        if use_tilda_embedding:
            with tf.variable_scope('', reuse=True):
                tilda_embeddings = tf.get_variable('tilda_embeddings')

        with tf.variable_scope(scope):
            with tf.variable_scope('embeddings'):

                if tilda_embeddings is not None:
                    embedding_table = tilda_embeddings
                else:
                    embedding_table = tf.get_variable(
                        name='word_embeddings',
                        shape=[vocab_size, embedding_size],
                        initializer=util.create_initializer(0.02),
                        dtype=tf.float32,
                        trainable=trainable)

                flat_input_ids = tf.reshape(input_ids, [-1])
                output = tf.gather(
                    embedding_table, flat_input_ids, name='embedding_look_up')
                output = tf.reshape(
                    output, [batch_size, max_seq_length, embedding_size])

                output_expanded = tf.expand_dims(output, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope('conv_%s' % filter_size):

                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_channels]
                    W = tf.get_variable(
                        name='W',
                        shape=filter_shape,
                        initializer=\
                            tf.truncated_normal_initializer(0.1),
                        dtype=tf.float32,
                        trainable=trainable)
                    b = tf.get_variable(
                        name='b',
                        shape=[num_channels],
                        initializer=\
                            tf.constant_initializer(0.1),
                        dtype=tf.float32,
                        trainable=trainable)
                    conv = tf.nn.conv2d(
                        output_expanded, W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, max_seq_length - int(filter_size) + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pooled)

            num_channels_total = num_channels * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [batch_size, num_channels_total])

            with tf.name_scope('dropout'):
                self.pooled_output = util.dropout(h_pool_flat, dropout_prob)

    def get_pooled_output(self):
        ''' Returns a tensor with shape [batch_size, hidden_size]. '''
        return self.pooled_output