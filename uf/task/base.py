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
''' Training methods including basic training and adversarial training. '''

import os
import time
import random
import numpy as np
import multiprocessing
from abc import abstractmethod

from ..tools import tf
from .. import utils


class BaseTask:
    ''' Parent class of all tasks.

    This is an internal class that does not provide interface
    for outside requests.'''

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_session(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.m._gpu_ids)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.m.sess = tf.Session(graph=self.m.graph, config=config)
        self._init_variables(self.m.global_variables)
        self.m._session_built = True

    def _init_variables(self, variables, ignore_checkpoint=False):

        tf.logging.info('Running local_init_op')
        local_init_op = tf.variables_initializer(variables)
        self.m.sess.run(local_init_op)
        self.m._inited_vars |= set(variables)
        tf.logging.info('Done running local_init_op')

        if not ignore_checkpoint and self.m.init_checkpoint:
            checkpoint_path = utils.get_checkpoint_path(self.m.init_checkpoint)
            if not checkpoint_path:
                raise ValueError('Checkpoint file \'%s\' does not exist. '
                                 'Make sure you pass correct value to '
                                 '`init_checkpoint`.'
                                 % self.m.init_checkpoint)
            self.m.init_checkpoint = checkpoint_path

            # `continual` means we tend to succeed the training step
            # and momentums variables in optimization
            continual = os.path.dirname(checkpoint_path) == self.m.output_dir
            if continual:
                self.m.step = int(checkpoint_path.split('-')[-1])

            (assignment_map, uninited_vars) = utils.get_assignment_map(
                checkpoint_path, variables, continual=continual)
            self.m.assignment_map = assignment_map
            self.m.uninited_vars = uninited_vars

            if uninited_vars:
                tf.logging.info(
                    '%d local variables failed to match up with the '
                    'checkpoint file. Check more details through '
                    '`.uninited_vars`.' % len(uninited_vars))

            if not self.m.assignment_map:    # no variables to restore
                return
            loader = tf.train.Saver(self.m.assignment_map)
            loader.restore(self.m.sess, checkpoint_path)

            if '_global_step' in self.m.__dict__:
                self.m.sess.run(tf.assign(self.m._global_step, self.m.step))

    def _build_feed_dict(self):
        feed_dict = {}
        for key, data in self.m.data.items():
            if key.startswith(utils.BACKUP_DATA):
                continue
            ptr = self._ptr
            batch = data[ptr: ptr + self.m.batch_size]
            ptr += self.m.batch_size
            while len(batch) < self.m.batch_size:
                ptr = self.m.batch_size - len(batch)
                remainder = data[:ptr]
                concat_func = np.vstack if len(batch.shape) > 1 else np.hstack
                batch = concat_func((batch, remainder))
            feed_dict[self.m.placeholders[key]] = batch
        self._ptr = ptr
        return feed_dict


class Training(BaseTask):

    def __init__(self, module, **kwargs):
        self.m = module
        self._kwargs = kwargs

        self.decorate()

    def decorate(self):
        self._set_placeholders()

        (grads, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(**self._kwargs)
        update_params_op = utils.update_global_params(
            self.m.trainable_variables, self.m._global_step, self.m._optimizer, grads)
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

    def run(self, target_steps,
            print_per_secs=60,
            save_per_steps=1000):
        adversarial = ''
        if self._kwargs.get('adversarial'):
            adversarial = self._kwargs.get('adversarial').lower()

        if self._kwargs.get('shuffle', True) and not self.tfrecords_files:
            self._shuffle()

        # init session
        if not self.m._session_built:
            utils.count_params(
                self.m.global_variables, self.m.trainable_variables)
            self._init_session()
        else:
            variables = []
            for var in self.m.global_variables:
                if var not in self.m._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables)
        self.m._session_mode = 'train'

        # print
        if adversarial:
            tf.logging.info(
                'Running adversarial training `%s` on %d samples (step %d -> %d)',
                adversarial, self.n_inputs, self.m.step, target_steps)
        else:
            tf.logging.info(
                'Running training on %d samples (step %d -> %d)',
                self.n_inputs, self.m.step, target_steps)

        # SMART: initialize tilda_embedding
        if adversarial == 'smart':
            self.m.sess.run(self.m._init_tilda_op)

        self._ptr = 0
        last_tic = time.time()
        last_step = self.m.step
        saver = tf.train.Saver(max_to_keep=self._kwargs.get('max_to_keep', 1000000))
        for _ in range(target_steps - self.m.step):
            last_tic, last_step = self._train_one_batch(
                self.m.step + 1, last_tic, last_step, target_steps,
                print_per_secs, save_per_steps, saver,
                adversarial)
            self.m.step += 1

    def _set_placeholders(self):
        self.from_tfrecords = bool(self._kwargs.get('tfrecords_files'))
        self.tfrecords_files = self._kwargs.get('tfrecords_files', [])
        self.n_jobs = self._kwargs.get('n_jobs')
        if self.from_tfrecords:
            self.n_inputs = utils.get_tfrecords_length(self.tfrecords_files)
            if self.n_jobs is None:
                self.n_jobs = max(multiprocessing.cpu_count() - 1, 1)

            self.m._set_placeholders('feature', is_training=True)
            features = {key: self.m.placeholders[key]
                        for key in utils.get_tfrecords_keys(
                            self.tfrecords_files[0])}

            def decode_record(record):
                example = tf.parse_single_example(
                    record, features)
                for name in list(example.keys()):
                    _t = example[name]
                    if _t.dtype == tf.int64:
                        _t = tf.to_int32(_t)
                    example[name] = _t
                return example

            dataset = tf.data.TFRecordDataset(self.tfrecords_files)
            dataset = dataset.repeat()
            if tf.__version__.startswith('1'):
                map_and_batch = tf.contrib.data.map_and_batch
            elif tf.__version__.startswith('2'):
                map_and_batch = tf.data.experimental.map_and_batch
            dataset = dataset.apply(map_and_batch(
                decode_record,
                batch_size=self.m.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            self.m.placeholders = iterator.get_next()
        else:
            self.n_inputs = len(list(self.m.data.values())[0])
            self.m._set_placeholders('placeholder', is_training=True)

        if not self.n_inputs:
            raise ValueError('0 input samples recognized.')

    def _shuffle(self):
        index_list = list(range(len(list(self.m.data.values())[0])))
        random.shuffle(index_list)
        for key, data in self.m.data.items():
            if key.startswith(utils.BACKUP_DATA):
                continue
            self.m.data[key] = self.m.data[key][index_list]

    def _train_one_batch(self, step, last_tic, last_step, target_steps,
                         print_per_secs, save_per_steps, saver,
                         adversarial=None):
        feed_dict = {}
        as_feature = True
        if not self.from_tfrecords:
            feed_dict = self._build_feed_dict()
            as_feature = False

        output_arrays = self.m.sess.run(self.m._get_fit_ops(as_feature), feed_dict=feed_dict)

        # print
        if time.time() - last_tic > print_per_secs or step == target_steps:
            info = 'step %d' % step

            # print processor-specific information
            info += self.m._get_fit_info(output_arrays, feed_dict, as_feature)

            # print training efficiency
            if time.time() - last_tic > print_per_secs or step == target_steps:
                info += ', %.2f steps/sec' % ((step - last_step) / (time.time() - last_tic))
                info += ', %.2f examples/sec' % ((step - last_step) / (time.time() - last_tic) * self.m.batch_size)

            tf.logging.info(info)
            last_tic = time.time()
            last_step = step

        # SMART: update tilda_embedding
        if step % self.m.steps_per_epoch == 0 and adversarial == 'smart':
            self.m.sess.run(self.m._update_tilda_op)

        # save
        if self.m.output_dir and step % save_per_steps == 0:
            tf.logging.info('Saving checkpoint for %d into %s/model.ckpt' % (step, self.m.output_dir))
            self.m.init_checkpoint = (self.m.output_dir + '/model.ckpt-%d' % step)
            saver.save(self.m.sess, self.m.init_checkpoint)

        return last_tic, last_step


class Inference(BaseTask):

    def __init__(self, module):
        self.m = module

        # ignore redundant building of the work flow
        if self.m._session_mode not in ('predict', 'score'):
            self.decorate()

    def decorate(self):
        self.m._set_placeholders('placeholder', is_training=False)

        (_, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(False)

    def run(self):
        n_inputs = len(list(self.m.data.values())[0])
        if not n_inputs:
            raise ValueError('0 input samples recognized.')

        # init session
        if not self.m._session_built:
            utils.count_params(self.m.global_variables, self.m.trainable_variables)
            self._init_session()
        self.m._session_mode = 'predict'

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.m.batch_size + 1
        for step in range(total_steps):
            self._predict_one_batch(step, last_tic, total_steps, batch_outputs)

        return self.m._get_predict_outputs(batch_outputs)

    def _predict_one_batch(self, step, last_tic,
                           total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        output_arrays = self.m.sess.run(self.m._get_predict_ops(), feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = 'Time usage %dm-%.2fs' % (diff_tic // 60, diff_tic % 60)
            info += ', %.2f steps/sec' % (total_steps / diff_tic)
            info += ', %.2f examples/sec' % (total_steps / diff_tic * self.m.batch_size)

            tf.logging.info(info)


class Scoring(BaseTask):

    def __init__(self, module):
        self.m = module

        # ignore redundant building of the work flow
        if self.m._session_mode not in ('predict', 'score'):
            self.decorate(self.m)

    def run(self):
        n_inputs = len(list(self.m.data.values())[0])
        if not n_inputs:
            raise ValueError('0 input samples recognized.')

        # init session
        if not self.m._session_built:
            utils.count_params(self.m.global_variables, self.m.trainable_variables)
            self._init_session()
        self.m._session_mode = 'score'

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.m.batch_size + 1
        for step in range(total_steps):
            self._score_one_batch(step, last_tic, total_steps, batch_outputs)

        return self.m._get_score_outputs(batch_outputs)

    def _score_one_batch(self, step, last_tic,
                         total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        output_arrays = self.m.sess.run(self.m._get_score_ops(), feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = 'Time usage %dm-%.2fs' % (diff_tic // 60, diff_tic % 60)
            info += ', %.2f steps/sec' % (total_steps / diff_tic)
            info += ', %.2f examples/sec' % (total_steps / diff_tic * self.m.batch_size)

            tf.logging.info(info)


class Initialization(BaseTask):

    def __init__(self, module):
        self.m = module

        self.decorate()

    def decorate(self):
        self.m._set_placeholders('placeholder', is_training=False)

        (_, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(False)

    def run(self, reinit_all, ignore_checkpoint):

        # init session
        if not self.m._session_built:
            utils.count_params(self.m.global_variables, self.m.trainable_variables)
            self._init_session(ignore_checkpoint)
        elif reinit_all:
            self._init_session(ignore_checkpoint)
        else:
            variables = []
            for var in self.m.global_variables:
                if var not in self.m._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables, ignore_checkpoint)
            else:
                tf.logging.info(
                    'Global variables already initialized. To '
                    're-initialize all, pass `reinit_all` to '
                    'True.')
        self.m._session_mode = 'predict'

    def _init_session(self, ignore_checkpoint):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.m._gpu_ids)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.m.sess = tf.Session(graph=self.m.graph, config=config)
        self._init_variables(self.m.global_variables, ignore_checkpoint)
        self.m._session_built = True


class Exportation(BaseTask):

    def __init__(self, module):
        self.m = module

        self.decorate()

    def decorate(self):
        self.m._set_placeholders('placeholder', on_export=True, is_training=False)

        (_, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(False)

    def run(self, export_dir, rename_inputs=None, rename_outputs=None,
            ignore_inputs=None, ignore_outputs=None):

        # init session
        if not self.m._session_built:
            utils.count_params(
                self.m.global_variables, self.m.trainable_variables)
            self._init_session()
        self.m._session_mode = 'predict'

        def set_input(key, value):
            inputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info('Register Input: %s, %s, %s' % (
                key, value.shape.as_list(), value.dtype.name))

        # define inputs
        inputs = {}
        if not ignore_inputs:
            ignore_inputs = []
        for key, value in list(self.m.placeholders.items()):
            if key == 'sample_weight' or key in ignore_inputs:
                continue
            if rename_inputs and key in rename_inputs:
                key = rename_inputs[key]
            set_input(key, value)

        def set_output(key, value):
            outputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info('Register Output: %s, %s, %s' % (
                key, value.shape.as_list(), value.dtype.name))

        # define outputs
        outputs = {}
        if not ignore_outputs:
            ignore_outputs = []
        for key, value in (list(self.m._preds.items()) +
                           list(self.m._probs.items())):
            if key in ignore_outputs:
                continue
            if rename_outputs and key in rename_outputs:
                key = rename_outputs[key]
            set_output(key, value)

        # build signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        signature_def_map = {'predict': signature}
        tf.logging.info('Register Signature: predict')

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = tf.saved_model.builder.SavedModelBuilder(
            os.path.join(export_dir, time.strftime('%Y%m%d.%H%M%S')))
        try:
            builder.add_meta_graph_and_variables(
                self.m.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                legacy_init_op=legacy_init_op)
        except Exception:
            raise ValueError(
                'Twice exportation is not allowed. Try `.save()` and '
                '`.reset()` method to save and reset the graph before '
                'next exportation.')
        builder.save()
