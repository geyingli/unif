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
''' Training methods including basic training and adversarial training. '''

import os
import time
import random
import numpy as np
from abc import abstractmethod

from .tools import tf, contrib
from . import utils


class BaseTask:
    ''' Parent class of `BasicTraining`, `AdversarialTraining`,
    `BasicInference`, `ExportInference` and `BasicScoring`.

    This is an internal class that does not provide interface
    for outside requests.'''

    # This method will be reimplemented by `BasicTraining` and
    # `AdversarialTraining` but not `BasicInference` and `BasicScoring`
    def __init__(self, module, **kwargs):
        self.module = module
        self.from_tfrecords = bool(kwargs.get('tfrecords_files'))
        self.tfrecords_files = kwargs.get('tfrecords_files', [])
        self.n_jobs = kwargs.get('n_jobs')
        self._kwargs = kwargs

        if self.from_tfrecords:
            self.n_inputs = utils.get_tfrecords_length(self.tfrecords_files)
        else:
            self.n_inputs = len(list(module.data.values())[0])

        # ignore redundant building of the work flow
        if module._graph_mode not in ('predict', 'score'):
            self.decorate(module)

    @abstractmethod
    def decorate(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_variables(self, variables):
        module = self.module

        tf.logging.info('Running local_init_op')
        local_init_op = tf.variables_initializer(variables)
        module.sess.run(local_init_op)
        module._inited_vars |= set(variables)
        tf.logging.info('Done running local_init_op')

        if module.init_checkpoint:
            checkpoint_path = utils.get_checkpoint_path(module.init_checkpoint)
            if not checkpoint_path:
                raise ValueError('Checkpoint file \'%s\' does not exist. '
                                 'Make sure you pass correct value to '
                                 '`init_checkpoint`.'
                                 % module.init_checkpoint)
            module.init_checkpoint = checkpoint_path
            continual = os.path.dirname(checkpoint_path) == module.output_dir
            if continual:
                module.step = int(checkpoint_path.split('-')[-1])

            (assignment_map, uninited_vars) = \
                utils.get_assignment_map(
                    checkpoint_path, variables, continual=continual)
            module.assignment_map = assignment_map
            module.uninited_vars = uninited_vars
            if not module.assignment_map:
                return
            loader = tf.train.Saver(module.assignment_map)
            loader.restore(module.sess, checkpoint_path)
            try:
                module.sess.run(tf.assign(module._global_step, module.step))
            except AttributeError:
                pass

    def _build_feed_dict(self):
        if self.from_tfrecords:
            return {}

        feed_dict = {}
        for key, data in self.module.data.items():
            ptr = self._ptr
            batch = data[ptr: ptr + self.module.batch_size]
            ptr += self.module.batch_size
            while len(batch) < self.module.batch_size:
                ptr = self.module.batch_size - len(batch)
                remainder = data[:ptr]
                concat_func = np.vstack if len(batch.shape) > 1 else np.hstack
                batch = concat_func((batch, remainder))
            feed_dict[self.module.placeholders[key]] = batch
        self._ptr = ptr
        return feed_dict


class BasicTraining(BaseTask):
    def __init__(self, module, **kwargs):
        self.module = module
        self.from_tfrecords = bool(kwargs.get('tfrecords_files'))
        self.tfrecords_files = kwargs.get('tfrecords_files', [])
        self.n_jobs = kwargs.get('n_jobs')
        self._kwargs = kwargs

        if self.from_tfrecords:
            self.n_inputs = utils.get_tfrecords_length(self.tfrecords_files)
        else:
            self.n_inputs = len(list(module.data.values())[0])

        self.decorate(module)

    def decorate(self, module):
        if self.from_tfrecords:
            module._set_placeholders('feature')
            features = {key: module.placeholders[key]
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
            dataset = dataset.apply(contrib.data.map_and_batch(
                decode_record,
                batch_size=module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            module.placeholders = iterator.get_next()
        else:
            module._set_placeholders('placeholder')

        (grads, module._losses, module._probs, module._preds) = \
            module._parallel_forward(**self._kwargs)
        update_params_op = utils.update_global_params(
            module.trainable_variables, module._global_step,
            module._optimizer, grads)
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

        if not module._graph_built:
            utils.count_params(
                module.global_variables, module.trainable_variables)

    def run(self, target_steps,
            print_per_secs=60,
            save_per_steps=1000):
        module = self.module
        shuffle = self._kwargs.get('shuffle', True)
        adversarial = ''
        if self._kwargs.get('adversarial'):
            adversarial = self._kwargs.get('adversarial').lower()

        if shuffle and not self.tfrecords_files:
            self._shuffle()
        self._ptr = 0
        saver = tf.train.Saver(
            max_to_keep=self._kwargs.get('max_to_keep', 1000000))

        if not module._graph_built:
            self._init_variables(module.global_variables)
        else:
            variables = []
            for var in module.global_variables:
                if var not in module._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables)

        # We cannot straightly infer since the dropout in training stage
        # creates fluctuations in outputs.
        module._graph_mode = 'train'
        module._graph_built = True

        # print
        if adversarial:
            tf.logging.info(
                'Running adversarial training `%s` on %d '
                'samples (step %d -> %d)',
                adversarial, self.n_inputs, module.step, target_steps)
        else:
            tf.logging.info(
                'Running training on %d samples (step %d -> %d)',
                self.n_inputs, module.step, target_steps)

        # SMART: initialize tilda_embedding
        if adversarial == 'smart':
            module.sess.run(module._init_tilda_op)

        last_tic = time.time()
        last_step = module.step
        for _ in range(target_steps - module.step):
            last_tic, last_step = self._train_one_batch(
                module.step + 1, last_tic, last_step, target_steps,
                print_per_secs, save_per_steps, saver,
                adversarial)
            module.step += 1

        if module.output_dir:
            tf.logging.info(
                'Saving checkpoint for %d into %s/model.ckpt'
                % (module.step, module.output_dir))
            module.init_checkpoint = \
                module.output_dir + '/model.ckpt-%d' % module.step
            saver.save(module.sess, module.init_checkpoint)

    def _shuffle(self):
        index_list = list(range(len(list(self.module.data.values())[0])))
        random.shuffle(index_list)
        for key, data in self.module.data.items():
            self.module.data[key] = self.module.data[key][index_list]

    def _train_one_batch(self, step, last_tic, last_step, target_steps,
                         print_per_secs, save_per_steps, saver,
                         adversarial=None):
        module = self.module
        feed_dict = self._build_feed_dict()
        as_feature = True if self.from_tfrecords else False

        output_arrays = module.sess.run(
            module._get_fit_ops(as_feature),
            feed_dict=feed_dict)

        # print
        if time.time() - last_tic > print_per_secs \
                or step == target_steps:
            info = 'step %d' % step

            # print processor-specific information
            info += module._get_fit_info(output_arrays, feed_dict, as_feature)

            # print training efficiency
            if time.time() - last_tic > print_per_secs \
                    or step == target_steps:
                info += ', %.2f steps/sec' % (
                    (step - last_step) / (
                        time.time() - last_tic))
                info += ', %.2f examples/sec' % (
                    (step - last_step) / (
                        time.time() - last_tic) * module.batch_size)

            tf.logging.info(info)
            last_tic = time.time()
            last_step = step

        # SMART: update tilda_embedding
        if step % module.steps_per_epoch == 0 and adversarial == 'smart':
            module.sess.run(module._update_tilda_op)

        # save
        if module.output_dir and step % save_per_steps == 0:
            tf.logging.info(
                'Saving checkpoint for %d into %s/model.ckpt'
                % (step, module.output_dir))
            module.init_checkpoint = (
                module.output_dir + '/model.ckpt-%d' % step)
            saver.save(module.sess, module.init_checkpoint)

        return last_tic, last_step


class AdversarialTraining(BasicTraining):
    def __init__(self, module, **kwargs):
        class_name = module.__class__.__name__
        if class_name.startswith('FastBERT'):
            raise ValueError(
                '%s does not support adversarial training.'
                % class_name)

        self.adversarial = kwargs.get('adversarial')
        super(AdversarialTraining, self).__init__(module, **kwargs)

    def decorate(self, module):
        if self.from_tfrecords:
            module._set_placeholders('feature')
            features = {key: module.placeholders[key]
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
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                decode_record,
                batch_size=module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            module.placeholders = iterator.get_next()
        else:
            module._set_placeholders('placeholder')

        adversarial = ''
        if self.adversarial:
            adversarial = self.adversarial.lower()

        if adversarial.lower() == 'fgm':
            self._fgm(module, **self._kwargs)
        elif adversarial.lower() == 'pgd':
            self._pgd(module, **self._kwargs)
        elif adversarial.lower() == 'freelb':
            self._freelb(module, **self._kwargs)
        elif adversarial.lower() == 'freeat':
            self._freeat(module, **self._kwargs)
        elif adversarial.lower() == 'smart':
            self._smart(module, **self._kwargs)
        else:
            raise ValueError(
                'Wrong adversarial algorithm <%s>. '
                'Pick one in the following list: '
                'FGM, PGD, FreeLB, FreeAT, SMART.' % self.adversarial)

        if not module._graph_built:
            utils.count_params(
                module.global_variables, module.trainable_variables)

    def _fgm(self, module, epsilon=0.5, **kwargs):
        # FGM takes average on actual gradient and virtual
        # gradient under attack.
        # i.e. grad = (actual_grad + last_grad) / 2
        #
        # The range of perturbation is fixed, which hardly reaches
        # optimized point. (epsilon: the range of perturbation over gradient,
        # must be smaller than one)

        # attack
        (actual_grads, module._losses, module._probs, module._preds) = \
            module._parallel_forward(**self._kwargs)
        grad, param = utils.get_grad_and_param(
            module.trainable_variables, actual_grads, 'word_embedding')
        r = tf.multiply(epsilon, grad / (tf.norm(grad) + 1e-9))
        attack_op = param.assign(param + r)

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = module._parallel_forward(**self._kwargs)
            restore_op = param.assign(param - r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(
                         actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            module.trainable_variables, module._global_step,
            module._optimizer, grads)
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

    def _pgd(self, module, epsilon=0.05, n_loop=2, **kwargs):
        # PGD takes average on actual gradient and last_tic gradient under
        # attack.
        # i.e. grad = (actual_grad + last_grad) / 2
        #
        # PGD adjusts perturbation through loop of attacks. Whenever
        # perturbation exceeds pre-assigned limit, it will be projected
        # to epsilon range. The perturbation is iterated.
        # (epsilon: the norm of perturbation, must be smaller than the
        # norm of gradients)

        # attack
        acc_r = 0.0
        attack_op = tf.no_op()
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                (d_grads, losses, probs, preds) = \
                    module._parallel_forward(**self._kwargs)
                if k == 0:
                    actual_grads = d_grads
                    (module._losses, module._probs, module._preds) = \
                        losses, probs, preds
                grad, param = utils.get_grad_and_param(
                    module.trainable_variables, d_grads, 'word_embedding')
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = (acc_r + tmp_r) * tf.divide(epsilon, norm)
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = module._parallel_forward(**self._kwargs)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(
                         actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            module.trainable_variables, module._global_step,
            module._optimizer, grads)
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

    def _freelb(self, module, epsilon=0.3, n_loop=3, **kwargs):
        # FreeLB is similar to PGD, but uses average gradients from loop.
        # i.e. grad = (first_grad + ... + last_grad) / n_loop
        #
        # Also, it initializes the perturbation not from usual forward
        # propagation, but a collection of uniform distribution within
        # epsilon range. It does not uses actual gradient to average
        # gradients. The perturbation is iterated, in the same way with
        #  PGD.
        # (epsilon: the norm of perturbation, must be smaller than the
        # norm of gradients)

        # initialize
        (d_grads, module._losses, module._probs, module._preds) = \
            module._parallel_forward(**self._kwargs)
        grad, param = utils.get_grad_and_param(
            module.trainable_variables, d_grads, 'word_embedding')
        init_r = tf.get_variable(
            'init_r',
            shape=[module.batch_size * module.max_seq_length,
                   param.shape.as_list()[-1]],
            initializer=tf.random_uniform_initializer(
                minval=-epsilon, maxval=epsilon),
            trainable=False)
        init_op = tf.variables_initializer([init_r])
        with tf.control_dependencies([init_op]):    # fix perturbation
            # Scale randomly initialized permutation, to make sure norm
            # of `r` is smaller than epsilon.
            shape = tf.cast(np.prod(init_r.shape.as_list()), tf.float32)
            r = tf.divide(init_r, tf.sqrt(shape))
            r = tf.IndexedSlices(values=r,
                                 indices=grad.indices,
                                 dense_shape=grad.dense_shape)
            attack_op = param.assign(param + r)

        with tf.control_dependencies([init_op]):    # fix perturbation
            # Scale randomly initialized permutation, to make sure norm
            # of `r` is smaller than epsilon.
            r = tf.divide(init_r, tf.norm(init_r, np.inf))
            r = tf.IndexedSlices(values=r,
                                 indices=grad.indices,
                                 dense_shape=grad.dense_shape)
            attack_op = param.assign(param + r)

        # attack
        acc_r = r
        all_grads = []
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                (attack_grads, _, _, _) = \
                    module._parallel_forward(**self._kwargs)
                all_grads.append(attack_grads)
                grad, _ = utils.get_grad_and_param(
                    module.trainable_variables,
                    attack_grads, 'word_embedding')
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = (acc_r + tmp_r) * tf.divide(epsilon, norm)
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = \
                module._parallel_forward(**self._kwargs)
            all_grads.append(attack_grads)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads(split_grad) for split_grad in zip(
                *all_grads)]
        update_params_op = utils.update_global_params(
            module.trainable_variables, module._global_step,
            module._optimizer, grads)
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

    def _freeat(self, module, epsilon=0.001, n_loop=3, **kwargs):
        # (epsilon: the range of perturbation over gradient,
        # must be smaller than one)

        # loop
        last_r = 0.0
        last_r_slice = 0.0
        attack_op = tf.no_op()
        for k in range(n_loop):

            # update
            with tf.control_dependencies([attack_op]):
                (grads, losses, probs, preds) = \
                    module._parallel_forward(**self._kwargs)
                if k == 0:
                    (module._losses, module._probs, module._preds) = \
                        losses, probs, preds
                grad, param = utils.get_grad_and_param(
                    module.trainable_variables, grads, 'word_embedding')
                update_params_op = utils.update_global_params(
                    module.trainable_variables, module._global_step,
                    module._optimizer, grads)

            # attack
            with tf.control_dependencies([update_params_op]):
                # any operator directly applied to `IndexedSlice` is dangerous
                values = grad.values
                sign = tf.cast(tf.greater(values, 0.0), tf.float32)
                r = last_r + tf.multiply(epsilon, sign) if k > 0 else \
                    tf.multiply(epsilon, sign)
                r *= tf.divide(epsilon, tf.norm(r))
                r_slice = tf.IndexedSlices(
                    values=r,
                    indices=grad.indices,
                    dense_shape=grad.dense_shape)
                attack_op = param.assign(param - last_r_slice + r_slice)
                last_r = r
                last_r_slice = r_slice
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

    def _smart(self, module, epsilon=0.01, n_loop=2,
               prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3,
               **kwargs):
        # SMART is essentially a different adversarial training algorithm
        # compared to the ones above. It consists of two key and new
        # features: smothness-inducing regularization and Bregman proximal
        # point optimization. Both the two features are directly reflected
        # in the loss function. When smoothness-inducing calculates
        # symmetrized KL-divergence between usual samples `x` and
        # perturbated samples `x+r`, Bregman proximal point optimization
        # punishes deviation from previous embeddings `tilda`.
        # (epsilon: the maxium norm of perturbation, must be smaller than
        # the largest value of gradients)

        # initialize
        (unused_grads, module._losses, module._probs, module._preds) = \
            module._parallel_forward(**self._kwargs)
        cls_loss = tf.reduce_mean(list(module._losses.values())[0])

        # Bregman proximal point optimization
        embedding_shape = utils.get_param(
            module.trainable_variables, 'word_embedding').shape.as_list()
        tilda = tf.get_variable(
            name='tilda_embeddings',
            shape=embedding_shape,
            initializer=tf.zeros_initializer, trainable=False)
        (_, _, breg_probs, _) = \
            module._parallel_forward(use_tilda_embedding=True, **self._kwargs)
        probs = list(module._probs.values())[0]
        probs_breg = list(breg_probs.values())[0]
        per_example_loss = tf.reduce_sum(
            probs_breg * (tf.log(probs_breg) - tf.log(probs)), axis=-1)
        per_example_loss_breg = tf.reduce_sum(
            probs * (tf.log(probs) - tf.log(probs_breg)), axis=-1)
        breg_loss = breg_miu * (
            tf.reduce_mean(per_example_loss) +
            tf.reduce_mean(per_example_loss_breg))
        module._losses['breg'] = breg_miu * (
            per_example_loss +
            per_example_loss_breg)

        # perturbation
        grad, param = utils.get_grad_and_param(
            module.trainable_variables, unused_grads, 'word_embedding')
        init_r = tf.get_variable(
            'init_r',
            shape=[module.batch_size * module.max_seq_length,
                   embedding_shape[-1]],
            initializer=tf.random_normal_initializer(stddev=epsilon),
            trainable=False)
        with tf.control_dependencies([breg_loss]):
            init_op = tf.variables_initializer([init_r])
        with tf.control_dependencies([init_op]):    # fix perturbation
            # Scale randomly initialized permutation, to make sure norm
            # of `r` is smaller than epsilon.
            r = tf.divide(init_r, tf.norm(init_r, np.inf))
            r = tf.IndexedSlices(values=r,
                                 indices=grad.indices,
                                 dense_shape=grad.dense_shape)
            attack_op = param.assign(param + r)

        # attack
        acc_r = r
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                (_, _, prtb_probs, _) = \
                    module._parallel_forward(**self._kwargs)

                # smoothness-inducing adversarial regulization
                probs_prtb = list(prtb_probs.values())[0]
                per_example_loss = tf.reduce_sum(
                    probs_prtb * (tf.log(probs_prtb) - tf.log(probs)), axis=-1)
                per_example_loss_prtb = tf.reduce_sum(
                    probs * (tf.log(probs) - tf.log(probs_prtb)), axis=-1)
                prtb_loss = prtb_lambda * (
                    tf.reduce_mean(per_example_loss) +
                    tf.reduce_mean(per_example_loss_prtb))
                module._losses['prtb'] = prtb_lambda * (
                    per_example_loss +
                    per_example_loss_prtb)

                # sum up
                total_loss = cls_loss + breg_loss + prtb_loss
                grads = tf.gradients(total_loss, module.trainable_variables)
                grad, _ = utils.get_grad_and_param(
                    module.trainable_variables, grads, 'word_embedding')

                tmp_r = tf.multiply(1 / n_loop, grad / (
                    tf.norm(grad, np.inf) + 1e-9))

                # In order not to shuffle the distribution of gradient-induced
                # perturbation, we use norm to scale instead of simply clip the
                # values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = (acc_r + tmp_r) * tf.divide(epsilon, norm)
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # update
        update_params_op = utils.update_global_params(
            module.trainable_variables, module._global_step,
            module._optimizer, grads)
        update_step_op = module._global_step.assign(module._global_step + 1)
        module._train_op = tf.group([update_params_op, update_step_op])

        # runs at the start of each epoch
        module._init_tilda_op = tilda.assign(param)

        # runs at the end of each epoch
        module._update_tilda_op = tilda.assign(
            (1 - tilda_beta) * param + tilda_beta * tilda)


class BasicInference(BaseTask):

    def decorate(self, module):
        if self.from_tfrecords:
            module._set_placeholders('feature')
            features = {key: module.placeholders[key]
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
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                decode_record,
                batch_size=module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            module.placeholders = iterator.get_next()
        else:
            module._set_placeholders('placeholder')

        (_, module._losses, module._probs, module._preds) = \
            module._parallel_forward(False, **self._kwargs)

        if not module._graph_built:
            utils.count_params(
                module.global_variables, module.trainable_variables)

    def run(self):
        module = self.module

        if not module._graph_built:
            self._init_variables(module.global_variables)

        # Now we can infer without rebuilding the graph, which
        # could be much faster.
        module._graph_mode = 'predict'
        module._graph_built = True

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (self.n_inputs - 1) // module.batch_size + 1
        for step in range(total_steps):
            self._predict_one_batch(
                step, last_tic, total_steps, batch_outputs)

        return module._get_predict_outputs(batch_outputs)

    def _predict_one_batch(self, step, last_tic,
                           total_steps, batch_outputs):
        module = self.module
        feed_dict = self._build_feed_dict()
        output_arrays = module.sess.run(
            module._get_predict_ops(), feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = 'Time usage %dm-%.2fs' % (diff_tic // 60, diff_tic % 60)
            info += ', %.2f steps/sec' % (total_steps / diff_tic)
            info += ', %.2f examples/sec' % (
                total_steps / diff_tic * module.batch_size)

            tf.logging.info(info)


class ExportInference(BaseTask):

    def decorate(self, module):
        module._set_placeholders('placeholder', on_export=True)

        (_, module._losses, module._probs, module._preds) = \
            module._parallel_forward(False, **self._kwargs)

        if not module._graph_built:
            utils.count_params(
                module.global_variables, module.trainable_variables)

    def run(self):
        module = self.module

        if not module._graph_built:
            self._init_variables(module.global_variables)

        # Now we can infer without rebuilding the graph, which
        # could be much faster.
        module._graph_mode = 'predict'
        module._graph_built = True

        # define inputs
        inputs = {}
        for key in module.placeholders:
            if key == 'sample_weight':
                continue
            inputs[key] = tf.saved_model.utils.build_tensor_info(
                module.placeholders[key])
            tf.logging.info(
                'Define input: %s, %s, %s'
                % (key, module.placeholders[key].shape.as_list(),
                   module.placeholders[key].dtype.name))

        # define outputs
        outputs = {}
        for p_id, (name, _probs) in enumerate(list(module._probs.items())):
            key = 'probabilities_%d' % name if p_id > 0 else 'probabilities'
            outputs[key] = tf.saved_model.utils.build_tensor_info(_probs)
            tf.logging.info(
                'Define output: %s, %s, %s'
                % (key, _probs.shape.as_list(), _probs.dtype.name))

        # build signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        signature_def_map = {'predict': signature}
        tf.logging.info('Define signature: predict')

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = tf.saved_model.builder.SavedModelBuilder(
            os.path.join(module.output_dir,
                         time.strftime('%Y%m%d.%H%M%S')))
        try:
            builder.add_meta_graph_and_variables(
                module.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                legacy_init_op=legacy_init_op)
        except:
            raise ValueError(
                'Twice exportation is not allowed.')
        builder.save()


class BasicScoring(BaseTask):

    def decorate(self, module):
        if self.from_tfrecords:
            module._set_placeholders('feature')
            features = {key: module.placeholders[key]
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
            dataset = dataset.apply(tf.contrib.data.map_and_batch(
                decode_record,
                batch_size=module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            module.placeholders = iterator.get_next()
        else:
            module._set_placeholders('placeholder')

        (_, module._losses, module._probs, module._preds) = \
            module._parallel_forward(False, **self._kwargs)

        if not module._graph_built:
            utils.count_params(
                module.global_variables, module.trainable_variables)

    def run(self):
        module = self.module
        n_inputs = len(list(self.module.data.values())[0])

        if not module._graph_built:
            self._init_variables(module.global_variables)

        # Now we can infer without rebuilding the graph, which
        # could be much faster.
        module._graph_mode = 'score'
        module._graph_built = True

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // module.batch_size + 1
        for step in range(total_steps):
            self._score_one_batch(
                step, last_tic, total_steps, batch_outputs)

        return module._get_score_outputs(batch_outputs)

    def _score_one_batch(self, step, last_tic,
                         total_steps, batch_outputs):
        module = self.module
        feed_dict = self._build_feed_dict()
        output_arrays = module.sess.run(
            module._get_score_ops(), feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = 'Time usage %dm-%.2fs' % (diff_tic // 60, diff_tic % 60)
            info += ', %.2f steps/sec' % (total_steps / diff_tic)
            info += ', %.2f examples/sec' % (
                total_steps / diff_tic * module.batch_size)

            tf.logging.info(info)
