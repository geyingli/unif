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
''' Optimization methods.
  Code revised from Google's implementation of BERT.
  See `https://github.com/google-research/bert`.
'''

import re

from .tools import tf


class UnifiedOptimizer:
    '''
    A unified optimizer for GD, Adam, AdamW and LAMB optimizers.
      - AdamW implementation is the implementation in BERT.
      - LAMB implementation is based on arXiv v3 version on June, 2019.
    '''
    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.98,
                 exclude_from_weight_decay=None,
                 layerwise_lr_decay=False,
                 optimizer='adamw',
                 **kwargs):
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = tf.cast(beta_1, dtype=tf.float32)
        self.beta_2 = tf.cast(beta_2, dtype=tf.float32)
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.layerwise_lr_decay = layerwise_lr_decay
        self.optimizer = optimizer.lower()
        self.kwargs = kwargs

        if self.optimizer in ('adam', 'adamw'):
            self.prefix = 'adam'
        elif self.optimizer == 'lamb':
            self.prefix = 'lamb'

    def _apply_gradients(self, grads_and_vars, learning_rate, global_step):
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            if self.optimizer == 'gd':
                update = grad

            else:
                param_name = self._get_variable_name(param.name)
                m = tf.get_variable(
                    name=param_name + '/%s_m' % self.prefix,
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                v = tf.get_variable(
                    name=param_name + '/%s_v' % self.prefix,
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

                next_m = (tf.multiply(self.beta_1, m) +
                          tf.multiply(1.0 - self.beta_1, grad))
                next_v = (tf.multiply(self.beta_2, v) +
                          tf.multiply(1.0 - self.beta_2, tf.square(grad)))

                # scaling
                update = next_m / (tf.sqrt(next_v) + 1e-6)

                # weight decay regularization
                if self.optimizer in ('adamw', 'lamb') and \
                        self._do_use_weight_decay(param_name):
                    update += self.weight_decay_rate * param

                # implemente lamb
                if self.optimizer == 'lamb':
                    # Note: Here are two choices for scaling function \phi(z)
                    # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
                    # identity: \phi(z) = z
                    # The authors does not mention what is \gamma_l and
                    # \gamma_u
                    r1 = tf.sqrt(tf.reduce_sum(tf.square(param)))
                    r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

                    r = tf.where(tf.greater(r1, 0.0),
                                 tf.where(tf.greater(r2, 0.0), r1 / r2, 1.0),
                                 1.0)
                    update *= r

                # update m, v
                assignments.extend([m.assign(next_m), v.assign(next_v)])

            next_param = param - learning_rate * update

            # update param
            assignments.extend([param.assign(next_param)])

        return assignments

    def apply_gradients(self, grads_and_vars, global_step, name=None):
        ''' Apply computed gradients to parameters. '''

        # layer-wise learning rate decay
        if self.layerwise_lr_decay:

            key_to_grads_and_vars = {}
            for grad, var in grads_and_vars:

                update_for_var = False
                for key in self.learning_rate:
                    if key in var.name:
                        update_for_var = True

                        if key not in key_to_grads_and_vars:
                            key_to_grads_and_vars[key] = []
                        key_to_grads_and_vars[key].append((grad, var))

                if not update_for_var:
                    raise ValueError(
                        'No learning rate specified for variable', var)

            assignments = []
            for key, key_grads_and_vars in key_to_grads_and_vars.items():
                assignments += self._apply_gradients(
                    key_grads_and_vars, self.learning_rate[key], global_step)
        else:
            assignments = self._apply_gradients(
                grads_and_vars, self.learning_rate, global_step)

        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for hint in self.exclude_from_weight_decay:
                if re.search(hint, param_name) is not None:
                    return False
        return True

    @staticmethod
    def _get_variable_name(param_name):
        res = re.match('^(.*):\\d+$', param_name)
        if res is not None:
            param_name = res.group(1)
        return param_name


def get_global_step():
    return tf.get_variable(
        'global_step', shape=(),
        initializer=tf.zeros_initializer,
        dtype=tf.int32,
        trainable=False)


def get_optimizer(init_lr, global_step, num_train_steps,
                  num_warmup_steps=None, key_to_depths=None, **kwargs):
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # learning rate linear decay
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # learning rate warmup
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
        global_steps_float = tf.cast(global_steps_int, dtype=tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, dtype=tf.float32)

        warmup_learning_rate = \
            init_lr * global_steps_float / warmup_steps_float
        is_warmup = tf.cast(
            global_steps_int < warmup_steps_int, dtype=tf.float32)
        learning_rate = ((1.0 - is_warmup) * learning_rate +
                         is_warmup * warmup_learning_rate)

    # layer-wise learning rate decay
    layerwise_lr_decay_ratio = kwargs.get('layerwise_lr_decay_ratio')
    if layerwise_lr_decay_ratio:
        if key_to_depths == 'unsupported':
            tf.logging.warning(
                'Layer-wise learning rate decay is not supported '
                'in the current module. Ignored.')
        else:
            learning_rate = {
                key: learning_rate * \
                    (layerwise_lr_decay_ratio ** depth)
                for (key, depth) in key_to_depths.items()}

    # optimier
    optimizer = UnifiedOptimizer(
        learning_rate=learning_rate,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'],
        layerwise_lr_decay=bool(layerwise_lr_decay_ratio),
        **kwargs)

    return optimizer
