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

import collections

from ..tools import tf

BACKUP_DATA = 'ex:'


def write_tfrecords(data, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    keys = []
    values = []
    for key, value in data.items():
        if key.startswith(BACKUP_DATA):
            continue
        keys.append(key)
        values.append(value)
    examples = zip(*values)

    for example in examples:
        features = collections.OrderedDict()
        for i, value in enumerate(example):
            if isinstance(value, int):
                features[keys[i]] = create_int_feature([value])
            elif isinstance(value, float):
                features[keys[i]] = create_float_feature([value])
            elif value.dtype.name.startswith('int'):
                features[keys[i]] = create_int_feature(value.tolist())
            elif value.dtype.name.startswith('float'):
                features[keys[i]] = create_float_feature(value.tolist())
            else:
                raise ValueError('Invalid data type: %s.' % type(value))
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def get_tfrecords_keys(tfrecords_file):
    iterator = tf.python_io.tf_record_iterator(tfrecords_file)
    record = next(iterator)
    example = tf.train.Example()
    example.ParseFromString(record)
    return list(example.features.feature.keys())


def get_tfrecords_length(tfrecords_files):
    n = 0
    for tfrecords_file in tfrecords_files:
        for _ in tf.python_io.tf_record_iterator(tfrecords_file):
            n += 1
    return n


def create_int_feature(values):
    ''' Convert list of values into tf-serializable Int64. '''
    if not isinstance(values, list):
        values = [values]
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=values))
    return feature


def create_float_feature(values):
    ''' Convert list of values into tf-serializable Float. '''
    if not isinstance(values, list):
        values = [values]
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=values))
    return feature
