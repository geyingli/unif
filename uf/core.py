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
''' Core methods and class. '''

import os
import json
import collections
from abc import abstractmethod

from .tools import tf
from .processing import (BasicTraining, AdversarialTraining,
                         BasicInference, ExportInference,
                         BasicScoring)
from . import optimization
from . import utils


class BaseModule:
    ''' Parent class of all the application processors. '''

    def __init__(self, init_checkpoint, output_dir, gpu_ids):

        # read checkpoint path
        self.init_checkpoint = init_checkpoint

        # create output directory
        self.output_dir = output_dir
        if output_dir:
            tf.gfile.MakeDirs(output_dir)
            tf.logging.info('Output directory: %s' % output_dir)

        # convert GPU ids to list
        self._gpu_ids = []
        if gpu_ids is None:
            gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        elif not gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if gpu_ids:
            try:
                if isinstance(gpu_ids, str):
                    self._gpu_ids = gpu_ids.replace(' ', '').split(',')
                else:
                    self._gpu_ids = list(map(str, gpu_ids))
            except Exception:
                raise ValueError(
                    '`gpu_ids` should be a list of GPU ids or a string '
                    'seperated with commas.')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(self._gpu_ids)

        # initialize graph and session
        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.InteractiveSession(
            graph=self.graph,
            config=config)

        # Before we build the graph, `score` and fast `predict`
        # is now allowed.
        self.step = 0
        self._graph_mode = None
        self._graph_built = False
        self._inited_vars = set()

    def __repr__(self):
        info = 'uf.' + self.__class__.__name__ + '('
        for key in self.__class__.__init__.__code__.co_varnames[1:]:
            try:
                value = self.__getattribute__(key)
            except:
                value = self.__init_args__[key]
            value = '\'%s\'' % value if isinstance(value, str) \
                else '%s' % value
            info += '%s=%s, ' % (key, value)
        return info[:-2] + ')'

    def __del__(self):
        try:
            self.sess.close()
        except Exception:
            pass

    def to_tfrecords(self, X=None, y=None, sample_weight=None,
                     X_tokenized=None, tfrecords_file=None):
        ''' Transform raw data and serialize into TFRecords.

        Args:
            X: list. A list object consisting untokenized inputs.
            y: list. A list object consisting labels.
            sample_weight: list. A list object of float-convertable values.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            tfrecords_file: string. The file path of TFRecords to write.
        Returns:
            None
        '''

        if not tfrecords_file:
            if self.output_dir:
                tfrecords_file = os.path.join(self.output_dir, '.tfrecords')
            else:
                tfrecords_file = '.tfrecords'

        data = self.convert(
            X, y, sample_weight, X_tokenized, is_training=True)
        utils.write_tfrecords(data, tfrecords_file)

    def fit_from_tfrecords(
            self, batch_size=32,
            learning_rate=5e-5,
            target_steps=None,
            total_steps=-3,
            warmup_ratio=0.1,
            print_per_secs=0.1,
            save_per_steps=1000,
            tfrecords_files=None,
            n_jobs=3,
            **kwargs):
        ''' Training the model using TFRecords.

        Args:
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            tfrecords_files: list. A list object of string defining TFRecords
              files to read.
            n_jobs: int. Number of threads in reading TFRecords files.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        '''

        # Make sure the arguments are correct.
        self.batch_size = batch_size
        if self._gpu_ids:
            assert batch_size % len(self._gpu_ids) == 0, (
                '`batch_size` should be evenly divided by the number of GPUs, '
                'but got %d and %d.'
                % (batch_size, len(self._gpu_ids)))

        # Get absolute path of tf.records file
        if not tfrecords_files:
            if self.output_dir:
                tfrecords_files = [os.path.join(self.output_dir, '.tfrecords')]
            else:
                tfrecords_files = ['.tfrecords']
        elif isinstance(tfrecords_files, str):
            tfrecords_files = tfrecords_files.split(',')

        # Confirm the number of training steps and warmup
        # steps. In reality, we use a slanted learning rate
        # that starts to decay after gradually climing to
        # the pre-assigned peak level.
        n_inputs = utils.get_tfrecords_length(tfrecords_files)
        self.steps_per_epoch = (n_inputs - 1) // batch_size + 1
        if total_steps < 0:
            total_steps = -total_steps * self.steps_per_epoch
        self.total_steps = int(total_steps)
        if not target_steps:
            target_steps = total_steps
        elif target_steps < 0:
            target_steps = int(-target_steps * self.steps_per_epoch)
        self.num_warmup_steps = int(total_steps * warmup_ratio)

        # Define optimization process, build the graph, and then run.
        with self.graph.as_default(), \
                tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self._global_step = optimization.get_global_step()
            self._optimizer = optimization.get_optimizer(
                init_lr=learning_rate,
                global_step=self._global_step,
                num_train_steps=self.total_steps,
                num_warmup_steps=self.num_warmup_steps,
                key_to_depths=self._key_to_depths,
                **kwargs)
            kwargs.update(tfrecords_files=tfrecords_files, n_jobs=n_jobs)
            return self._build('fit', **kwargs).run(
                target_steps,
                print_per_secs=print_per_secs,
                save_per_steps=save_per_steps)

    def fit(self, X=None, y=None, sample_weight=None, X_tokenized=None,
            batch_size=32,
            learning_rate=5e-5,
            target_steps=None,
            total_steps=-3,
            warmup_ratio=0.1,
            print_per_secs=0.1,
            save_per_steps=1000,
            **kwargs):
        ''' Training the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            y: list. A list object consisting labels.
            sample_weight: list. A list object of float-convertable values.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            learning_rate: float. Peak learning rate during training process.
            target_steps: float/int. The number of target steps, must be
              smaller or equal to `total_steps`. When assigned to a negative
              value, the model automatically calculate the required steps to
              finish a loop which covers all training data, then the value is
              multiplied with the absolute value of `target_steps` to obtain
              the real target number of steps.
            total_steps: int. The number of total steps in optimization, must
              be larger or equal to `target_steps`. When assigned to a
              negative value, the model automatically calculate the required
              steps to finish a loop which covers all training data, then the
              value is multiplied with the absolute value of `total_steps` to
              obtain the real number of total steps.
            warmup_ratio: float. How much percentage of total steps fall into
              warming up stage.
            print_per_secs: int. How many steps to print training information,
              e.g. training loss.
            save_per_steps: int. How many steps to save model into checkpoint
              file. Valid only when `output_dir` is not None.
            **kwargs: Other arguments about layer-wise learning rate decay,
              adversarial training or model-specific settings. See `README.md`
              to obtain more
        Returns:
            None
        '''

        # Make sure the arguments are correct.
        self.batch_size = batch_size
        if self._gpu_ids:
            assert batch_size % len(self._gpu_ids) == 0, (
                '`batch_size` should be evenly divided by the number of GPUs, '
                'but got %d and %d.'
                % (batch_size, len(self._gpu_ids)))

        # Convert raw data to structed data. This method
        # should be specifically implemented by child classes.
        self.data = self.convert(
            X, y, sample_weight, X_tokenized, is_training=True)

        # Confirm the number of training steps and warmup
        # steps. In reality, we use a slanted learning rate
        # that starts to decay after gradually climing to
        # the pre-assigned peak level.
        n_inputs = len(list(self.data.values())[0])
        self.steps_per_epoch = (n_inputs - 1) // batch_size + 1
        if total_steps < 0:
            total_steps = -total_steps * self.steps_per_epoch
        self.total_steps = total_steps
        if not target_steps:
            target_steps = total_steps
        elif target_steps < 0:
            target_steps = -target_steps * self.steps_per_epoch
        self.num_warmup_steps = int(total_steps * warmup_ratio)

        # Define optimization process, build the graph, and then run.
        with self.graph.as_default(), \
                tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self._global_step = optimization.get_global_step()
            self._optimizer = optimization.get_optimizer(
                init_lr=learning_rate,
                global_step=self._global_step,
                num_train_steps=self.total_steps,
                num_warmup_steps=self.num_warmup_steps,
                key_to_depths=self._key_to_depths,
                **kwargs)
            return self._build('fit', **kwargs).run(
                target_steps,
                print_per_secs=print_per_secs,
                save_per_steps=save_per_steps)

    def predict(self, X=None, X_tokenized=None, batch_size=8):
        ''' Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
        Returns:
            A dict object of model outputs.
        '''

        # Make sure the arguments are correct.
        self.batch_size = batch_size
        if self._gpu_ids:
            assert batch_size % len(self._gpu_ids) == 0, (
                '`batch_size` should be evenly divided by the number of GPUs, '
                'but got %d and %d.'
                % (batch_size, len(self._gpu_ids)))

        # Make sure necessary arguments are on spot.
        if not self._graph_built:
            _attr_dict = self.__class__._INFER_ATTRIBUTES
            _miss_dict = set()
            for attr in _attr_dict:
                if self.__getattribute__(attr) is None:
                    _miss_dict.add(attr)
            if _miss_dict:
                _miss_info = []
                for attr in _miss_dict:
                    _miss_info += ['`%s`: %s' % (attr, _attr_dict[attr])]
                raise ValueError(
                    'Train the model first or feed value for the '
                    'following necessary arguments before running '
                    'inference. (%s)'
                    % '; '.join(_miss_info))

        # Convert raw data to structed data. This method
        # should be specifically implemented by child classes.
        self.data = self.convert(
            X, None, None, X_tokenized, is_training=False)

        # Build the graph, and then run.
        with self.graph.as_default(), \
                tf.variable_scope('', reuse=tf.AUTO_REUSE):
            return self._build('predict').run()

    def score(self, X=None, y=None, sample_weight=None, X_tokenized=None,
              batch_size=8):
        ''' Inference on the model with scoring.

        Args:
            X: list. A list object consisting untokenized inputs.
            y: list. A list object consisting labels.
            sample_weight: list. A list object of float-convertable values.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
        Returns:
            A dict object of output metrics.
        '''
        assert y is not None, '`y` can\'t be None.'

        # Make sure the arguments are correct.
        self.batch_size = batch_size
        if self._gpu_ids:
            assert batch_size % len(self._gpu_ids) == 0, (
                '`batch_size` should be evenly divided by the number of GPUs, '
                'but got %d and %d.'
                % (batch_size, len(self._gpu_ids)))

        # Make sure necessary arguments are on spot.
        if not self._graph_built:
            _attr_dict = self.__class__._INFER_ATTRIBUTES
            _miss_dict = set()
            for attr in _attr_dict:
                if self.__getattribute__(attr) is None:
                    _miss_dict.add(attr)
            if _miss_dict:
                _miss_info = []
                for attr in _miss_dict:
                    _miss_info += ['`%s`: %s' % (attr, _attr_dict[attr])]
                raise ValueError(
                    'Train the model first or feed value for the '
                    'following necessary arguments before running '
                    'scoring. (%s)'
                    % '; '.join(_miss_info))

        # Convert raw data to structed data. This method
        # should be specifically implemented by child classes.
        self.data = self.convert(
            X, y, sample_weight, X_tokenized, is_training=False)

        # Build the graph, and then run.
        with self.graph.as_default(), \
                tf.variable_scope('', reuse=tf.AUTO_REUSE):
            return self._build('score').run()

    def save(self):
        ''' Save model into checkpoint file.

        When attribute `output_dir` is None, the method is illegal. Otherwise
        the model will be saved into `"model.checkpoint-%s" % step` under
        the directory of `output_dir`.
        '''
        if not self._graph_built:
            raise ValueError(
                'Fit, predict or score before saving checkpoint.')

        if not self.output_dir:
            raise ValueError('Attribute `output_dir` is None.')

        tf.logging.info(
            'Saving checkpoint for %d into %s/model.ckpt'
            % (self.step, self.output_dir))
        self.init_checkpoint = (
            self.output_dir + '/model.ckpt-%d' % self.step)

        saver = tf.train.Saver(max_to_keep=1000000)
        saver.save(self.sess, self.init_checkpoint)

    def cache(self, code, cache_file='./.cache'):
        ''' Save model configurations into cache file.

        Args:
            code: string. Unique name of configuration to save. Can be any
              kind of string.
            cache_file: string. The path of cache file.
        Returns:
            None

        When attribute `output_dir` is not None, the method will save the
        model into checkpoint file simultaneously.
        '''
        if self.output_dir and self._graph_built:
            self.save()

        if os.path.exists(cache_file):
            cache_fp = open(cache_file, encoding='utf-8')
            cache_json = json.load(cache_fp)
            cache_fp.close()
        else:
            cache_json = {}

        _cache_json = {'keys': [], 'values': []}
        for key in self.__class__.__init__.__code__.co_varnames[1:]:
            try:
                value = self.__getattribute__(key)
            except:
                value = self.__init_args__[key]
            _cache_json['keys'].append(key)
            _cache_json['values'].append(value)
        cache_json[code] = _cache_json

        cache_fp = open(cache_file, 'w', encoding='utf-8')
        json.dump(cache_json, cache_fp, indent=2)
        cache_fp.close()

    @classmethod
    def load(cls, code, cache_file='./.cache'):
        ''' Load model from configurations saved in cache file.

        Args:
            code: string. Unique name of configuration to load.
            cache_file: string. The path of cache file.
        Returns:
            None
        '''
        if not os.path.exists(cache_file):
            raise ValueError('No cache file found with `%s`.' % cache_file)
        cache_fp = open(cache_file, encoding='utf-8')
        cache_json = json.load(cache_fp)
        cache_fp.close()

        if code not in cache_json.keys():
            raise ValueError(
                'No cached configs found with `code = %s`.' % code)
        args = collections.OrderedDict()
        for key, value in zip(
                cache_json[code]['keys'], cache_json[code]['values']):
            args[key] = value
        return cls(**args)

    def reinit_from_checkpoint(self, init_checkpoint=None,
                               assignment_map=None):
        ''' Reinitialize variables from checkpoint file.

        Args:
            init_checkpoint: string. Path of checkpoint file from which to
              load.
            assignment_map: dict. A dict object that maps from variable name
              in checkpoint to variables in local graph.
        '''
        if not self._graph_built:
            raise ValueError(
                'Fit, predict or score before running reinialization.')

        if not init_checkpoint:
            init_checkpoint = self.init_checkpoint
        checkpoint_path = utils.get_checkpoint_path(init_checkpoint)
        if not checkpoint_path:
            raise ValueError('Checkpoint file \'%s\' does not exist. '
                             'Make sure you pass correct value to '
                             '`init_checkpoint`.' % init_checkpoint)
        self.init_checkpoint = checkpoint_path

        continual = os.path.dirname(checkpoint_path) == self.output_dir
        if continual:
            self.step = int(checkpoint_path.split('-')[-1])

        # Add new trainable variables into assignment_map
        if not assignment_map:
            (assignment_map, _) = utils.get_assignment_map(
                checkpoint_path, self.global_variables, continual=False)
            for key in assignment_map:
                if key not in self.assignment_map:
                    self.assignment_map[key] = assignment_map[key]

        self.assignment_map.update(assignment_map)
        loader = tf.train.Saver(self.assignment_map)
        loader.restore(self.sess, checkpoint_path)
        try:
            self.sess.run(tf.assign(self._global_step, self.step))
        except AttributeError:
            pass

        new_uninited_vars = {}
        for var in self.global_variables:
            if var not in self.assignment_map.values():
                new_uninited_vars[var.name[:-2]] = var
        self.uninited_vars = new_uninited_vars

    @property
    def trainable_variables(self):
        return self.graph._collections.get('trainable_variables', [])

    @property
    def global_variables(self):
        return self.graph._collections.get('variables', [])

    def export(self):
        ''' Export model into SavedModel files.

        The method is illegal if attribute `output_dir` is None.
        '''
        if not self.output_dir:
            raise ValueError('Attribute `output_dir` is None.')

        # Make sure necessary arguments are on spot.
        if not self._graph_built:
            _attr_dict = self.__class__._INFER_ATTRIBUTES
            _miss_dict = set()
            for attr in _attr_dict:
                if self.__getattribute__(attr) is None:
                    _miss_dict.add(attr)
            if _miss_dict:
                _miss_info = []
                for attr in _miss_dict:
                    _miss_info += ['`%s`: %s' % (attr, _attr_dict[attr])]
                raise ValueError(
                    'Feed value for the following necessary arguments '
                    'before exportation of PB files. (%s)'
                    % '; '.join(_miss_info))

        # Build the graph, and then run.
        with self.graph.as_default(), \
                tf.variable_scope('', reuse=tf.AUTO_REUSE):
            self._build('export').run()

    @abstractmethod
    def convert(self, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def _convert_sample_weight(sample_weight, n_inputs):
        ''' Standardize `sample_weight`. '''
        if sample_weight:
            try:
                return [float(item) for item in sample_weight]
            except ValueError:
                raise ValueError(
                    '`sample_weight` must be a list of float-'
                    'convertable values.')
        return [1.0 for _ in range(n_inputs)]

    @staticmethod
    def _assert_legal(X, y, sample_weight, X_tokenized):
        ''' Make sure strange errors intrigged by data not occur. '''

        if X:
            if X_tokenized:
                raise ValueError('Set None to one of `X` and `X_tokenized`.')
            assert isinstance(X, list), '`X` should be a list object.'
        else:
            if not X_tokenized:
                raise ValueError('Must pass value to `X` or `X_tokenized`.')
            assert isinstance(X_tokenized, list), (
                '`X_tokenized` should be a list object.')
            X = X_tokenized

        if y:
            assert isinstance(y, list), '`y` should be a list object.'
            assert len(X) == len(y), (
                'Length of `y` should be the same with `X/X_tokenized`. '
                '(%d vs. %d)' % (len(y), len(X)))

        if sample_weight:
            assert isinstance(sample_weight, list), (
                '`sample_weight` should be a list object.')
            assert len(X) == len(sample_weight), (
                'Length of `sample_weight` should be the '
                'same with `X/X_tokenized`. (%d vs. %d)' % (len(y), len(X)))

    def _build(self, work, **kwargs):
        ''' Build the computation graph. '''

        # Build work flow with computation graph. Multi-GPU
        # training and inference are supported. Temporarily
        # not support running on TPUs.
        if work == 'fit':
            if kwargs.get('adversarial'):
                return AdversarialTraining(self, **kwargs)
            else:
                return BasicTraining(self, **kwargs)
        elif work == 'predict':
            return BasicInference(self, **kwargs)
        elif work == 'score':
            return BasicScoring(self, **kwargs)
        elif work == 'export':
            return ExportInference(self, **kwargs)

    @abstractmethod
    def _set_placeholders(self, *args, **kwargs):
        raise NotImplementedError()

    def _parallel_forward(self, is_training=True, **kwargs):
        ''' Parallel foundation of computation graph in multi-GPUs,
        a general method. '''

        # We implement data parallelization instead of model
        # parallelization, for this design suits most real cases.
        all_grads = []
        all_losses = []
        all_probs = []
        all_preds = []
        n_device = len(self._gpu_ids) if self._gpu_ids else 1
        split_placeholders = {key: {} for key in range(n_device)}
        for name, placeholder in self.placeholders.items():
            split_placeholder = tf.split(placeholder, n_device, axis=0)
            for key in range(n_device):
                split_placeholders[key][name] = split_placeholder[key]

        # map
        # The `Null` class makes the following codes about running on GPUs
        # compatible with running on CPU.
        device = utils.Null if n_device <= 1 else tf.device
        for idx in range(n_device):
            _gpu_id = self._gpu_ids[idx] if self._gpu_ids else ''
            with device('gpu:%s' % _gpu_id):
                (total_loss, d_losses, d_probs, d_preds) = self._forward(
                    is_training=is_training,
                    split_placeholders=split_placeholders[idx],
                    **kwargs)

                if is_training:
                    # This is the so-called 'backward' process
                    d_grads = tf.gradients(
                        total_loss, self.trainable_variables)
                    all_grads.append(d_grads)

                all_losses.append(d_losses)
                all_probs.append(d_probs)
                all_preds.append(d_preds)

        # reduce
        losses = collections.OrderedDict()
        probs = collections.OrderedDict()
        preds = collections.OrderedDict()
        for key in d_losses:
            _losses = [d_losses[key] for d_losses in all_losses]
            losses[key] = tf.concat(_losses, axis=0)
        for key in d_probs:
            _probs = [d_probs[key] for d_probs in all_probs]
            probs[key] = tf.concat(_probs, axis=0)
        for key in d_preds:
            _preds = [d_preds[key] for d_preds in all_preds]
            preds[key] = tf.concat(_preds, axis=0)

        # average, clip, and apply gradients
        grads = None
        if is_training:

            # average gradients
            # This process can be generalized to one device, so we do not
            # add another `if` expression.
            average_grads = []
            for i in range(len(self.trainable_variables)):
                split_grads = []
                for d_grads in all_grads:
                    if d_grads[i] is not None:
                        split_grads.append(d_grads[i])
                if split_grads:
                    average_grad = utils.average_n_grads(split_grads)
                    average_grads.append(average_grad)
                else:
                    average_grads.append(None)

            # clip gradients
            (grads, _) = tf.clip_by_global_norm(average_grads, clip_norm=1.0)

        return (grads, losses, probs, preds)

    @abstractmethod
    def _forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_fit_ops(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_fit_info(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_predict_ops(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_predict_outputs(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_score_ops(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _get_score_outputs(self, *args, **kwargs):
        raise NotImplementedError()
