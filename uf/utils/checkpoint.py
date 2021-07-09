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

import os
import re

from ..tools import tf


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
