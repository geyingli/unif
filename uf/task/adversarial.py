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

import numpy as np

from ..tools import tf
from .. import utils
from .base import Training


class AdversarialTraining(Training):

    def __init__(self, module, **kwargs):
        self.adversarial = kwargs.get('adversarial', '')
        super(AdversarialTraining, self).__init__(module, **kwargs)

    def decorate(self):
        self._set_placeholders()

        self.adversarial = self.adversarial.lower()
        try:
            ok = True
            if self.adversarial == 'fgm':
                self._fgm(**self._kwargs)
            elif self.adversarial == 'pgd':
                self._pgd(**self._kwargs)
            elif self.adversarial == 'freelb':
                self._freelb(**self._kwargs)
            elif self.adversarial == 'freeat':
                self._freeat(**self._kwargs)
            elif self.adversarial == 'smart':
                self._smart(**self._kwargs)
            else:
                ok = False
        except Exception:
            raise ValueError('`%s` does not support adversarial training.'
                             % self.m.__class__.__name__)
        if not ok:
            raise ValueError(
                'Wrong adversarial algorithm `%s`. '
                'Pick one in the following list: '
                'FGM, PGD, FreeLB, FreeAT, SMART.' % self.adversarial)

    def _fgm(self, epsilon=0.5, **kwargs):
        # FGM takes average on actual gradient and virtual
        # gradient under attack.
        # i.e. grad = (actual_grad + last_grad) / 2
        #
        # The range of perturbation is fixed, which hardly reaches
        # optimized point. (epsilon: the range of perturbation over gradient,
        # must be smaller than one)

        # attack
        (actual_grads, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(**self._kwargs)
        grad, param = utils.get_grad_and_param(self.m.trainable_variables, actual_grads, 'word_embedding')
        r = tf.multiply(epsilon, grad / (tf.norm(grad) + 1e-9))
        attack_op = param.assign(param + r)

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = self.m._parallel_forward(**self._kwargs)
            restore_op = param.assign(param - r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            self.m.trainable_variables, self.m._global_step,
            self.m._optimizer, grads)
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

    def _pgd(self, epsilon=0.05, n_loop=2, **kwargs):
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
                (d_grads, losses, probs, preds) = self.m._parallel_forward(**self._kwargs)
                if k == 0:
                    actual_grads = d_grads
                    (self.m._losses, self.m._probs, self.m._preds) = losses, probs, preds
                grad, param = utils.get_grad_and_param(self.m.trainable_variables, d_grads, 'word_embedding')
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = tf.cond(
                    norm > epsilon,
                    lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                    lambda: (acc_r + tmp_r))
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = self.m._parallel_forward(**self._kwargs)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(
                         actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            self.m.trainable_variables, self.m._global_step,
            self.m._optimizer, grads)
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

    def _freelb(self, epsilon=0.3, n_loop=3, **kwargs):
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
        (d_grads, self.m._losses, self.m._probs, self.m._preds) = self.m._parallel_forward(**self._kwargs)
        grad, param = utils.get_grad_and_param(self.m.trainable_variables, d_grads, 'word_embedding')
        init_r = tf.get_variable(
            'init_r',
            shape=[self.m.batch_size * self.m.max_seq_length,
                   param.shape.as_list()[-1]],
            initializer=tf.random_uniform_initializer(
                minval=-epsilon, maxval=epsilon),
            trainable=False)
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
        all_grads = []
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                (attack_grads, _, _, _) = \
                    self.m._parallel_forward(**self._kwargs)
                all_grads.append(attack_grads)
                grad, _ = utils.get_grad_and_param(
                    self.m.trainable_variables,
                    attack_grads, 'word_embedding')
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = tf.cond(
                    norm > epsilon,
                    lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                    lambda: (acc_r + tmp_r))
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # restore
        with tf.control_dependencies([attack_op]):
            (attack_grads, _, _, _) = \
                self.m._parallel_forward(**self._kwargs)
            all_grads.append(attack_grads)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads(split_grad) for split_grad in zip(
                *all_grads)]
        update_params_op = utils.update_global_params(
            self.m.trainable_variables, self.m._global_step,
            self.m._optimizer, grads)
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

    def _freeat(self, epsilon=0.001, n_loop=3, **kwargs):
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
                    self.m._parallel_forward(**self._kwargs)
                if k == 0:
                    (self.m._losses, self.m._probs, self.m._preds) = \
                        losses, probs, preds
                grad, param = utils.get_grad_and_param(
                    self.m.trainable_variables, grads, 'word_embedding')
                update_params_op = utils.update_global_params(
                    self.m.trainable_variables, self.m._global_step,
                    self.m._optimizer, grads)

            # attack
            with tf.control_dependencies([update_params_op]):
                # any operator directly applied to `IndexedSlice` is dangerous
                values = grad.values
                sign = tf.cast(tf.greater(values, 0.0), tf.float32)
                r = last_r + tf.multiply(epsilon, sign) if k > 0 else \
                    tf.multiply(epsilon, sign)
                r = tf.cond(tf.norm(r) > epsilon,
                            lambda: r * tf.divide(epsilon, tf.norm(r)),
                            lambda: r)
                r_slice = tf.IndexedSlices(
                    values=r,
                    indices=grad.indices,
                    dense_shape=grad.dense_shape)
                attack_op = param.assign(param - last_r_slice + r_slice)
                last_r = r
                last_r_slice = r_slice
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

    def _smart(self, epsilon=0.01, n_loop=2,
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
        (unused_grads, self.m._losses, self.m._probs, self.m._preds) = \
            self.m._parallel_forward(**self._kwargs)
        cls_loss = tf.reduce_mean(list(self.m._losses.values())[0])

        # Bregman proximal point optimization
        param = utils.get_param(self.m.trainable_variables, 'word_embedding')
        embedding_shape = param.shape.as_list()
        tilda = tf.get_variable(
            name='tilda_embeddings',
            shape=embedding_shape,
            initializer=tf.zeros_initializer, trainable=False)
        (_, _, breg_probs, _) = \
            self.m._parallel_forward(use_tilda_embedding=True, **self._kwargs)
        probs = list(self.m._probs.values())[0]
        probs_breg = list(breg_probs.values())[0]
        per_example_loss = tf.reduce_sum(
            probs_breg * (tf.log(probs_breg) - tf.log(probs)), axis=-1)
        per_example_loss_breg = tf.reduce_sum(
            probs * (tf.log(probs) - tf.log(probs_breg)), axis=-1)
        breg_loss = breg_miu * (
            tf.reduce_mean(per_example_loss) +
            tf.reduce_mean(per_example_loss_breg))
        self.m._losses['breg'] = breg_miu * (
            per_example_loss +
            per_example_loss_breg)

        # perturbation
        grad, param = utils.get_grad_and_param(
            self.m.trainable_variables, unused_grads, 'word_embedding')
        init_r = tf.get_variable(
            'init_r',
            shape=[self.m.batch_size * self.m.max_seq_length,
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
                    self.m._parallel_forward(**self._kwargs)

                # smoothness-inducing adversarial regulization
                probs_prtb = list(prtb_probs.values())[0]
                per_example_loss = tf.reduce_sum(
                    probs_prtb * (tf.log(probs_prtb) - tf.log(probs)), axis=-1)
                per_example_loss_prtb = tf.reduce_sum(
                    probs * (tf.log(probs) - tf.log(probs_prtb)), axis=-1)
                prtb_loss = prtb_lambda * (
                    tf.reduce_mean(per_example_loss) +
                    tf.reduce_mean(per_example_loss_prtb))
                self.m._losses['prtb'] = prtb_lambda * (
                    per_example_loss +
                    per_example_loss_prtb)

                # sum up
                total_loss = cls_loss + breg_loss + prtb_loss
                grads = tf.gradients(total_loss, self.m.trainable_variables)
                grad, _ = utils.get_grad_and_param(
                    self.m.trainable_variables, grads, 'word_embedding')

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
            self.m.trainable_variables, self.m._global_step,
            self.m._optimizer, grads)
        update_step_op = self.m._global_step.assign(self.m._global_step + 1)
        self.m._train_op = tf.group([update_params_op, update_step_op])

        # runs at the start of each epoch
        self.m._init_tilda_op = tilda.assign(param)

        # runs at the end of each epoch
        self.m._update_tilda_op = tilda.assign(
            (1 - tilda_beta) * param + tilda_beta * tilda)
