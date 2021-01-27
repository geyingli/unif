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
""" Useful methods to support applications. """

import os
import re
import json
import logging
import collections
import numpy as np

from .tools import tf
from uf.tokenization.word_piece import (
    _is_whitespace as is_whitespace,
    _is_punctuation as is_punctuation,
    _is_chinese_char as is_chinese_char,
    )
from . import application

PACK_DIR = os.path.dirname(__file__)
SIGNS = ('~`!@#$%^&*()_-+=][\}{|;\':",./><?·！@#￥%……&*（）——+'
         '【】「」：；”    “’‘《，》。？、')



class Null:
    ''' A null class for keeping code compatible when hanging out. '''
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def unimported_module(name, required):
    class UnimportedModule:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'Module `%s` is required to launch %s. Check '
                '"https://pypi.org/project/%s/" for installation details.'
                % (required, name, required))
    return UnimportedModule


class TFModuleError(Exception):
    def __init__(self, *args, **kwargs):
        pass


def load(code, cache_file='./.cache', **kwargs):
    ''' Load model from configurations saved in cache file.

    Args:
        code: string. Unique name of configuration to load.
        cache_file: string. The path of cache file.
    Returns:
        None
    '''
    tf.logging.info('Loading model `%s` from %s' % (code, cache_file))

    if not os.path.exists(cache_file):
        raise ValueError('No cache file found with `%s`.' % cache_file)
    cache_fp = open(cache_file, encoding='utf-8')
    cache_json = json.load(cache_fp)
    cache_fp.close()

    if code not in cache_json.keys():
        raise ValueError(
            'No cached configs found with code `%s`.' % code)
    if 'model' not in cache_json[code]:
        raise ValueError(
            'No model assigned. Try `uf.XXX.load()` instead.')
    model = cache_json[code]['model']
    args = collections.OrderedDict()

    # unif >= beta v2.1.35
    if '__init__' in cache_json[code]:
        zips = cache_json[code]['__init__'].items()
    # unif < beta v2.1.35
    elif 'keys' in cache_json[code]:
        zips = zip(cache_json[code]['keys'], cache_json[code]['values'])
    else:
        raise ValueError('Wrong format of cache file.')

    cache_dir = os.path.dirname(cache_file)
    if cache_dir == '':
        cache_dir = '.'
    for key, value in zips:

        # convert from relative path
        if key == 'init_checkpoint' or key.endswith('_dir') or \
                key.endswith('_file'):
            if isinstance(value, str) and not value.startswith('/'):
                value = get_simplified_path(
                    cache_dir + '/' + value)

        if key in kwargs:
            value = kwargs[key]
        args[key] = value
    return application.__dict__[model](**args)


def set_verbosity(level=2):
    if level == 2:
        tf.logging.set_verbosity(tf.logging.INFO)
    elif level == 1:
        tf.logging.set_verbosity(tf.logging.WARN)
    elif level == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        raise ValueError(
          'Invalid value: %s. Pick from `0`, `1` and `2`. '
          'The larger the value, the more information will be printed.'
          % level)


def set_log(log_file):
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    log.addHandler(fh)


def write_tfrecords(data, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    data_items = list(data.items())
    data_keys = [item[0] for item in data_items]
    data_values = [item[1] for item in data_items]

    for example in zip(*data_values):
        features = collections.OrderedDict()
        for i, value in enumerate(example):
            if isinstance(value, int):
                features[data_keys[i]] = create_int_feature([value])
            elif isinstance(value, float):
                features[data_keys[i]] = create_float_feature([value])
            elif value.dtype.name.startswith('int'):
                features[data_keys[i]] = create_int_feature(value.tolist())
            elif value.dtype.name.startswith('float'):
                features[data_keys[i]] = create_float_feature(value.tolist())
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


def get_relative_path(source, target):
    source = source.replace('\\', '/')
    target = target.replace('\\', '/')

    if source.startswith('/'):
        raise ValueError('Not a relative path: %s.' % source)
    if target.startswith('/'):
        raise ValueError('Not a relative path: %s.' % target)

    output = get_reverse_path(source) + '/' + target
    output = get_simplified_path(output)
    return output


def get_simplified_path(path):
    path = path.replace('\\', '/')
    while True:
        res = re.findall('[^/]+/../', path)
        res = [item for item in res if item != '../../' and item != './../']
        if res:
            path = path.replace(res[0], '')
        else:
            return path.replace('/./', '/')


def get_reverse_path(path):
    path = path.replace('\\', '/')

    if path.startswith('/'):
        raise ValueError('Not a relative path.')

    output = ''

    if os.path.isdir(path):
        if path.endswith('/'):
            path = path[:-1]
    else:
        path = os.path.dirname(path)

    if path == '':
        return '.'

    cwd = os.getcwd()
    for seg in path.split('/'):
        if seg == '.':
            pass
        elif seg == '..':
            output = '/' + cwd.split('/')[-1] + output
            cwd = os.path.dirname(cwd)
        else:
            output = '/..' + output
            cwd += '/' + seg

    output = output[1:]

    if output == '':
        return '.'

    return output


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


def get_placeholder(target, name, shape, dtype):
    if target == 'placeholder':
        return tf.placeholder(name=name, shape=shape, dtype=dtype)

    if dtype.name.startswith('int'):
        dtype = tf.int64
    elif dtype.name.startswith('int'):
        dtype = tf.float32
    return tf.FixedLenFeature(shape[1:], dtype)


def get_checkpoint_path(ckpt_dir):
    ''' If detected no checkpoint file, return None. '''
    if os.path.isdir(ckpt_dir):
        if not os.path.exists(os.path.join(ckpt_dir, 'checkpoint')):
            for file in os.listdir(ckpt_dir):
                if file.endswith('.index'):
                    return os.path.join(ckpt_dir, file.replace('.index', ''))
            return None
        with open(os.path.join(ckpt_dir, 'checkpoint')) as f:
            line = f.readline()
        try:
            file = re.findall('model_checkpoint_path: "(.+?)"', line)[0]
        except IndexError:
            return None
        if os.path.exists(os.path.join(ckpt_dir, file + '.index')):
            return os.path.join(ckpt_dir, file)
    else:
        if os.path.isfile(ckpt_dir + '.index'):
            return ckpt_dir
        filename = ckpt_dir.split('/')[-1]
        dirname = os.path.dirname(ckpt_dir)
        if not dirname:
            dirname = '.'

        exists_similar = False
        for file in os.listdir(dirname):
            try:
                re.findall(r'%s-[\d]+[.]index' % filename, file)
            except IndexError:
                continue
            if re.findall(r'%s-[\d]+[.]index' % filename, file):
                exists_similar = True
                break
        if not exists_similar:
            return None
        if not os.path.exists(os.path.join(dirname, 'checkpoint')):
            return None
        with open(os.path.join(dirname, 'checkpoint')) as f:
            line = f.readline()
        try:
            file = re.findall('model_checkpoint_path: "(.+?)"', line)[0]
        except IndexError:
            return None
        if os.path.exists(os.path.join(dirname, file + '.index')):
            return os.path.join(dirname, file)
    return None


def get_assignment_map(checkpoint_file,
                       variables,
                       continual=False,
                       show_matched=False):
    ''' Carefully designed so as to fulfil any personalized needs. '''
    assignment_map = {}

    # read local variables
    name_to_variable = {}
    for var in variables:
        name = var.name
        res = re.match('^(.*):\\d+$', name)
        if res is not None:
            name = res.group(1)
        if not continual:
            if 'global_step' in name \
                    or '/adam' in name \
                    or '/Adam' in name \
                    or '/lamb' in name:
                continue
        name_to_variable[name] = var

    # read checkpoint variables
    init_vars = tf.train.list_variables(checkpoint_file)
    inited_vars = {}
    for name_shape in init_vars:
        (from_name, from_shape) = (name_shape[0], name_shape[1])

        to_name = from_name
        if to_name not in name_to_variable or \
                name_to_variable[to_name].shape.as_list() != from_shape:
            if show_matched:
                tf.logging.info('checkpoint_file contains <%s>', from_name)
            continue
        if show_matched:
            tf.logging.info('checkpoint_file contains <%s>, matched', from_name)
        assignment_map[from_name] = name_to_variable[to_name]
        inited_vars[to_name] = 1

    # further feedback
    uninited_vars = {}
    for var in variables:
        if var.name[:-2] not in inited_vars:
            if var.name[:-2].endswith('_m') or var.name[:-2].endswith('_v'):
                continue
            if show_matched:
                tf.logging.info('unmatched parameter %s', var)
            uninited_vars[var.name[:-2]] = var
    return (assignment_map, uninited_vars)


def list_variables(checkpoint):
    checkpoint_path = get_checkpoint_path(checkpoint)
    if not checkpoint_path:
        raise ValueError('Checkpoint file \'%s\' does not exist. '
                         'Make sure you pass correct value to '
                         '`checkpoint`.' % checkpoint)
    return tf.train.list_variables(checkpoint_path)


def get_grad_and_param(variables, grads, param_name):
    for (grad, param) in zip(grads, variables):
        if param_name in param.name:
            return (grad, param)


def get_param(variables, param_name):
    for param in variables:
        if param_name in param.name:
            return param


def count_params(global_variables, trainable_variables):
    def get_params(variable):
        _tuple = tuple(map(int, variable.shape))
        if not _tuple:
            return 0
        return np.prod(_tuple)
    n_global = 0
    for variable in global_variables:
        n_global += get_params(variable)
    n_trainable = 0
    for variable in trainable_variables:
        n_trainable += get_params(variable)
    tf.logging.info('Build graph with %s parameters '
                    '(among which %s are trainable)'
                    % (format(int(n_global), ','),
                       format(int(n_trainable), ',')))


def average_n_grads(split_grads):
    # Dealing with IndexedSlices for large-dimensional embedding
    # matrix. The gradient of an embedding matrix is not a tensor,
    # but a tuple-like object named `IndexedSlices`, for this one,
    # we need to take special processings.
    if split_grads[0].__str__().startswith('IndexedSlices'):
        all_values = [grad.values for grad in split_grads]

        values = tf.divide(tf.add_n(all_values), len(split_grads))
        indices = split_grads[0].indices
        dense_shape = split_grads[0].dense_shape

        return tf.IndexedSlices(
            values=values,
            indices=indices,
            dense_shape=dense_shape)
    return tf.divide(tf.add_n(split_grads), len(split_grads))


def update_global_params(variables, global_step, optimizer, grads):
    update_op = optimizer.apply_gradients(
        zip(grads, variables), global_step=global_step)
    return tf.group(update_op)


def find_boyer_moore(T, P, start=0):
    ''' BM algorithm for string match. '''

    n, m = len(T), len(P)
    last = {}
    for k in range(m):
        last[P[k]] = k

    # align end of pattern at index m-1 of text
    i = start + m - 1
    k = m - 1
    while i < n:
        if T[i] == P[k]:
            if k == 0:
                return i
            i -= 1
            k -= 1
        else:
            j = last.get(T[i], -1)
            i += m - min(k, j + 1)
            k = m - 1
    return -1


def find_all_boyer_moore(T, P):
    start_ids = []
    start = 0
    while True:
        start_position = find_boyer_moore(
            T, P, start=start)
        if start_position == -1:
            break
        start_ids.append(start_position)
        start = start_position + len(P)
    return start_ids


def is_english_char(char):
    if re.findall('[a-zA-Z]', char):
        return True
    return False


def is_numeric_char(char):
    if re.findall('[\d]', char):
        return True
    return False


def convert_tokens_to_text(tokens):
    words = ['']
    for _token in tokens:
        if _token.startswith('##'):
            words[-1] += _token[2:]
        else:
            words.append(_token)
    text = ' '.join(words)

    # remove spaces
    if len(text) >= 3:
        i = 1
        while i < len(text) - 1:
            if is_whitespace(text[i]):
                _last = text[i - 1]
                _next = text[i + 1]

                # remove space between chars and punctuations
                if not is_english_char(_last) or not is_english_char(_next):
                    text = text.replace(
                        '%s%s%s' % (_last, text[i], _next),
                        '%s%s' % (_last, _next))
            i += 1

    return text.strip()


def align_tokens_with_text(tokens, text, lower_case):
    if lower_case:
        text = text.lower()

    i = 0
    j = 0
    max_j = len(text)
    mapping_start = []
    mapping_end = []
    while i < len(tokens):
        token = tokens[i]
        token = token.replace('##', '')
        if text[j:].startswith(token):
            mapping_start.append(j)
            mapping_end.append(j + len(token))
            i += 1
            j += len(token)
        elif token not in text[j:]:  # [CLS], [SEP], some Japanese signs
            mapping_start.append(j)
            if token in ('[CLS]', '[SEP]'):
                mapping_end.append(j)
            else:
                mapping_end.append(j + len(token))
            i += 1
        else:
            j += 1
        if j >= max_j:
            break

    for _ in range(len(tokens) - len(mapping_start)):
        mapping_start.append(max_j + 1000)
        mapping_end.append(max_j + 1000)

    return mapping_start, mapping_end


def transform(output_arrays, n_inputs, reshape=False):
    if len(output_arrays[0].shape) > 1:
        return np.vstack(output_arrays)[:n_inputs]
    if reshape:
        return np.vstack(np.reshape(
            output_arrays, [n_inputs, -1]))[:n_inputs]
    return np.hstack(output_arrays)[:n_inputs]


def truncate_segments(segments, max_seq_length, truncate_method='LIFO'):
    total_seq_length = sum([len(segment) for segment in segments])
    if total_seq_length <= max_seq_length:
        return

    for _ in range(total_seq_length - max_seq_length):
        if truncate_method == 'longer-FO':
            max(segments, key=lambda x: len(x)).pop()
        elif truncate_method == 'FIFO':
            index = 0
            while len(segments[index]) == 0:
                index += 1
            segments[index].pop(0)
        elif truncate_method == 'LIFO':
            index = -1
            while len(segments[index]) == 0:
                index -= 1
            segments[index].pop()
        else:
            raise ValueError(
                'Invalid value for `truncate_method`. '
                'Pick one from `longer-FO`, `FIFO`, `LIFO`.')
