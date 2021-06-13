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
""" Useful methods to support applications. """

import os
import re
import json
import time
import copy
import logging
try:
    import requests
except:
    pass
import collections
import numpy as np
import multiprocessing
from sys import stdout

from .tools import tf
from uf.tokenization.word_piece import (
    _is_whitespace as is_whitespace,
    _is_punctuation as is_punctuation,
    _is_chinese_char as is_chinese_char,
    )
from . import application

PACK_DIR = os.path.dirname(__file__)



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


def warning(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            tf.logging.warning(e)
    return wrapper


NUM_PROCESSES = 1
pool = None


class MultiProcess:
    def __init__(self, n_process='auto'):
        n_cpu = multiprocessing.cpu_count()
        if n_process != 'auto':
            assert n_process <= n_cpu, (
                'Invalid value of `n_process`. It can not exceed the num '
                'of cpu cores in the device: %d.' % n_cpu)
        else:
            n_process = n_cpu
        self.n = n_process

    def __enter__(self):
        global NUM_PROCESSES, pool
        if self.n > 1:
            pool = multiprocessing.Pool(self.n)
        NUM_PROCESSES = self.n

    def __exit__(self, *args, **kwargs):
        global NUM_PROCESSES, pool
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        NUM_PROCESSES = 1


def _parallel_convert_single_process(args):
    bucket_id = args[0]
    app_class = args[1]
    mapping = args[2]
    data = args[3]
    is_training = args[4]

    # Verbosity of tensorflow in new process will be set to default,
    # for this reason we just have to silence the logging and don't
    # have to care about the recovery.
    tf.logging.set_verbosity(tf.logging.FATAL)
    model = app_class(*mapping)

    data = model.convert(
        data['X'], data['y'], data['sample_weight'], data['X_tokenized'],
        is_training)
    return (bucket_id, data)


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


RESOURCES = [
    ['bert-base-zh', 'BERT', 'Google',
     'https://github.com/google-research/bert',
     'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip'],

    ['albert-tiny-zh', 'ALBERT', 'Google',
     'https://github.com/google-research/albert',
     'https://storage.googleapis.com/albert_zh/albert_tiny_zh_google.zip'],
    ['albert-small-zh', 'ALBERT', 'Google',
     'https://github.com/google-research/albert',
     'https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip'],

    ['albert-base-zh', 'ALBERT', 'Brightmart',
     'https://github.com/brightmart/albert_zh',
     'https://storage.googleapis.com/albert_zh/albert_base_zh_additional_36k_steps.zip'],
    ['albert-large-zh', 'ALBERT', 'Brightmart',
     'https://github.com/brightmart/albert_zh',
     'https://storage.googleapis.com/albert_zh/albert_large_zh.zip'],
    ['albert-xlarge-zh', 'ALBERT', 'Brightmart',
     'https://github.com/brightmart/albert_zh',
     'https://storage.googleapis.com/albert_zh/albert_xlarge_zh_183k.zip'],

    ['bert-wwm-ext-base-zh', 'BERT', 'HFL',
     'https://github.com/ymcui/Chinese-BERT-wwm',
     'https://drive.google.com/uc?export=download&id=1buMLEjdtrXE2c4G1rpsNGWEx7lUQ0RHi'],
    ['roberta-wwm-ext-base-zh', 'BERT', 'HFL',
     'https://github.com/ymcui/Chinese-BERT-wwm',
     'https://drive.google.com/uc?export=download&id=1jMAKIJmPn7kADgD3yQZhpsqM-IRM1qZt'],
    ['roberta-wwm-ext-large-zh', 'BERT', 'HFL',
     'https://github.com/ymcui/Chinese-BERT-wwm',
     'https://drive.google.com/uc?export=download&id=1dtad0FFzG11CBsawu8hvwwzU2R0FDI94'],

    ['macbert-base-zh', 'BERT', 'HFL',
     'https://github.com/ymcui/MacBERT',
     'https://drive.google.com/uc?export=download&id=1aV69OhYzIwj_hn-kO1RiBa-m8QAusQ5b'],
    ['macbert-large-zh', 'BERT', 'HFL',
     'https://github.com/ymcui/MacBERT',
     'https://drive.google.com/uc?export=download&id=1lWYxnk1EqTA2Q20_IShxBrCPc5VSDCkT'],

    ['xlnet-base-zh', 'XLNet', 'HFL',
     'https://github.com/ymcui/Chinese-XLNet',
     'https://drive.google.com/uc?export=download&id=1m9t-a4gKimbkP5rqGXXsEAEPhJSZ8tvx'],
    ['xlnet-mid-zh', 'XLNet', 'HFL',
     'https://github.com/ymcui/Chinese-XLNet',
     'https://drive.google.com/uc?export=download&id=1342uBc7ZmQwV6Hm6eUIN_OnBSz1LcvfA'],

    ['electra-180g-small-zh', 'ELECTRA', 'HFL',
     'https://github.com/ymcui/Chinese-ELECTRA',
     'https://drive.google.com/uc?export=download&id=177EVNTQpH2BRW-35-0LNLjV86MuDnEmu'],
    ['electra-180g-small-ex-zh', 'ELECTRA', 'HFL',
     'https://github.com/ymcui/Chinese-ELECTRA',
     'https://drive.google.com/uc?export=download&id=1NYJTKH1dWzrIBi86VSUK-Ml9Dsso_kuf'],
    ['electra-180g-base-zh', 'ELECTRA', 'HFL',
     'https://github.com/ymcui/Chinese-ELECTRA',
     'https://drive.google.com/uc?export=download&id=1RlmfBgyEwKVBFagafYvJgyCGuj7cTHfh'],
    ['electra-180g-large-zh', 'ELECTRA', 'HFL',
     'https://github.com/ymcui/Chinese-ELECTRA',
     'https://drive.google.com/uc?export=download&id=1P9yAuW0-HR7WvZ2r2weTnx3slo6f5u9q'],
]


def list_resources():
    columns = ['Key', 'Backbone', 'Organization', 'Site', 'URL']
    lengths = [len(col) for col in columns]
    resources = copy.deepcopy(RESOURCES)

    # scan for maximum length
    for i in range(len(resources)):
        for j in range(len(columns)):
            lengths[j] = max(lengths[j], len(resources[i][j]))
    seps = ['─' * length for length in lengths]

    # re-scan to modify length
    tf.logging.info('┌─' + '─┬─'.join(seps) + '─┐')
    for j in range(len(columns)):
        columns[j] += ' ' * (lengths[j] - len(columns[j]))
    tf.logging.info('┊ ' + ' ┊ '.join(columns) + ' ┊')
    tf.logging.info('├─' + '─┼─'.join(seps) + '─┤')

    for i in range(len(resources)):
        for j in range(len(columns)):
            resources[i][j] += ' ' * (lengths[j] - len(resources[i][j]))
        tf.logging.info('┊ ' + ' ┊ '.join(resources[i]) + ' ┊')
    tf.logging.info('└─' + '─┴─'.join(seps) + '─┘')


def download(key):
    resources = {item[0]: item for item in RESOURCES}
    if key not in resources:
        raise ValueError('Invalid key: %s. Check available resources '
                         'through `uf.list_resources()`.' % key)
    url = resources[key][-1]

    # download files
    try:
        if 'drive.google.com' in url:
            path = get_download_path(key, '.tar.gz')
            download_from_google_drive(url, path)
        elif 'storage.googleapis.com' in url:
            path = get_download_path(key, '.zip')
            download_from_google_apis(url, path)
    except KeyboardInterrupt:
        os.remove(path)


def download_all():
    resources = {item[0]: item for item in RESOURCES}
    for key in resources:
        url = resources[key][-1]

        try:
            if 'drive.google.com' in url:
                path = get_download_path(key, '.tar.gz')
                download_from_google_drive(url, path)
            elif 'storage.googleapis.com' in url:
                path = get_download_path(key, '.zip')
                download_from_google_apis(url, path)
        except KeyboardInterrupt:
            os.remove(path)
            return


def download_from_google_drive(url, path):
    with open(path, 'wb') as writer:
        file_id = url.split('id=')[-1]
        ori_url = url
        url = 'https://docs.google.com/uc?export=download'

        session = requests.Session()
        r = session.get(url, params={'id': file_id}, stream=True)
        def _get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = _get_confirm_token(r)

        if token:
            params = {'id': file_id, 'confirm': token}
            r = session.get(url, params=params, stream=True)
        tf.logging.info('Downloading files from %s' % (ori_url))

        chunk_size = 10240
        acc_size = 0
        cur_size = 0
        speed = None
        last_tic = time.time()
        for chunk in r.iter_content(chunk_size):
            if not chunk:
                continue

            writer.write(chunk)
            cur_size += chunk_size
            acc_size += chunk_size
            tic = time.time()

            if tic - last_tic > 3 or speed is None:
                span = tic - last_tic
                speed = '%.2fMB/s' % (cur_size / span / (1024 ** 2))
                last_tic = tic
                cur_size = 0
            stdout.write(
                'Downloading ... %.2fMB [%s] \r'
                % (acc_size / (1024 ** 2), speed))

        # extract files
        # extract_dir = path.replace('.tar.gz', '')
        # with tarfile.open(path) as tar_file:
        #     tar_file.extractall(extract_dir)
        # os.remove(path)
        tf.logging.info(
            'Succesfully downloaded. Saved into ./%s' % path)


def download_from_google_apis(url, path):
    with requests.get(url, stream=True) as r, open(path, 'wb') as writer:
        file_size = int(r.headers['Content-Length'])
        tf.logging.info('Downloading files from %s (%dMB)'
                        % (url, file_size // (1024 ** 2)))

        chunk_size = 10240
        percentage = 0
        cur_size = 0
        percentage_step = chunk_size / file_size
        speed = None
        last_tic = time.time()
        for chunk in r.iter_content(chunk_size):
            if not chunk:
                continue

            writer.write(chunk)
            cur_size += chunk_size
            percentage += percentage_step
            percentage = min(percentage, 1.0)
            tic = time.time()

            if tic - last_tic > 3 or speed is None:
                span = tic - last_tic
                speed = '%.2fMB/s' % (cur_size / span / (1024 ** 2))
                last_tic = tic
                cur_size = 0
            stdout.write(
                'Downloading ... %.2f%% [%s] \r' % (percentage * 100, speed))

        # extract files
        # extract_dir = path.replace('.zip', '')
        # tf.gfile.MakeDirs(extract_dir)
        # with zipfile.ZipFile(path) as zip_file:
        #     zip_file.extractall(extract_dir)
        # os.remove(path)
        tf.logging.info(
            'Succesfully downloaded. Saved into ./%s' % path)


def get_download_path(key, suffix='.zip'):
    new_path = key + suffix
    if not os.path.exists(new_path):
        return new_path
    index = 1
    while True:
        new_path = key + ' (%d)' % index + suffix
        if not os.path.exists(new_path):
            return new_path
        index += 1


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


def write_tfrecords_multi_process(data, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    data_items = list(data.items())
    data_keys = [item[0] for item in data_items]
    data_values = [item[1] for item in data_items]
    examples = list(zip(*data_values))

    NUM_BUCKETS = 20
    buckets = [[] for _ in range(NUM_BUCKETS)]
    n = len(examples)
    while n > 0:
        bucket_id = n % NUM_BUCKETS
        buckets[bucket_id].append(examples.pop())
        n -= 1

    n_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cpu)
    args = zip(buckets, [data_keys for _ in range(NUM_BUCKETS)])
    features_lists = pool.map(write_tfrecords_single_process, args)

    for features_list in features_lists:
        for features in features_list:
            tf_example = tf.train.Example(
                features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())


def write_tfrecords_single_process(args):
    examples = args[0]
    data_keys = args[1]
    features_list = []
    for example in examples:
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
        features_list.append(features)
    return features_list


def write_tfrecords(data, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    data_items = list(data.items())
    data_keys = [item[0] for item in data_items]
    data_values = [item[1] for item in data_items]
    examples = zip(*data_values)

    for example in examples:
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


def get_init_values(model):
    values = []
    for key in model.__class__.__init__.__code__.co_varnames[1:]:
        try:
            value = model.__getattribute__(key)
        except:
            value = model.__init_args__[key]
        values.append(value)
    return values


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
        res = re.findall('[^/]+/[.][.]/', path)
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
    split_grads = [grad for grad in split_grads if grad is not None]
    if len(split_grads) == 1:
        return split_grads[0]

    # Dealing with IndexedSlices for large-dimensional embedding
    # matrix. The gradient of an embedding matrix is not a tensor,
    # but a tuple-like object named `IndexedSlices`, for this one,
    # we need to take special processings.
    if split_grads[0].__str__().startswith('IndexedSlices'):

        values = tf.concat([grad.values for grad in split_grads], axis=0)
        indices = tf.concat([grad.indices for grad in split_grads], axis=0)
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


def transform(output_arrays, n_inputs=None, reshape=False):
    if not n_inputs:
        n_inputs = 100000000
    if len(output_arrays[0].shape) > 1:
        return np.vstack(output_arrays)[:n_inputs]
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
