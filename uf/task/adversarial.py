import numpy as np

from ..tools import tf
from .. import utils
from .base import Training


class AdversarialTraining(Training):

    def decorate(self, **kwargs):
        self._set_placeholders()

        try:
            ok = True
            if self.adversarial == "fgm":
                self._fgm(**kwargs)
            elif self.adversarial == "pgd":
                self._pgd(**kwargs)
            elif self.adversarial == "freelb":
                self._freelb(**kwargs)
            elif self.adversarial == "freeat":
                self._freeat(**kwargs)
            elif self.adversarial == "smart":
                self._smart(**kwargs)
            else:
                ok = False
        except Exception:
            raise ValueError("`%s` does not support adversarial training."
                             % self.module.__class__.__name__)
        if not ok:
            raise ValueError(
                "Wrong adversarial algorithm `%s`. "
                "Pick one in the following list: "
                "FGM, PGD, FreeLB, FreeAT, SMART." % self.adversarial)

    def _fgm(self, epsilon=0.5, **kwargs):
        # FGM takes average on actual gradient and virtual
        # gradient under attack.
        # i.e. grad = (actual_grad + last_grad) / 2
        #
        # The range of perturbation is fixed, which hardly reaches
        # optimized point. (epsilon: the range of perturbation over gradient,
        # must be smaller than one)

        # attack
        actual_grads, self.module._tensors = self.module._parallel_forward(**kwargs)
        grad, param = utils.get_grad_and_param(self.module.trainable_variables, actual_grads, "word_embedding")
        r = tf.multiply(epsilon, grad / (tf.norm(grad) + 1e-9))
        attack_op = param.assign(param + r)

        # restore
        with tf.control_dependencies([attack_op]):
            attack_grads, _ = self.module._parallel_forward(**kwargs)
            restore_op = param.assign(param - r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            self.module.trainable_variables, self.module._global_step,
            self.module._optimizer, grads)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

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
                d_grads, tensors = self.module._parallel_forward(**kwargs)
                if k == 0:
                    actual_grads = d_grads
                    self.module._tensors = tensors
                grad, param = utils.get_grad_and_param(self.module.trainable_variables, d_grads, "word_embedding")
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
            attack_grads, _ = self.module._parallel_forward(**kwargs)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads([actual_grad, attack_grad])
                     for (actual_grad, attack_grad) in zip(
                         actual_grads, attack_grads)]
        update_params_op = utils.update_global_params(
            self.module.trainable_variables, self.module._global_step,
            self.module._optimizer, grads)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

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
        d_grads, self.module._tensors = self.module._parallel_forward(**kwargs)
        grad, param = utils.get_grad_and_param(self.module.trainable_variables, d_grads, "word_embedding")
        init_r = tf.get_variable(
            "init_r",
            shape=[self.module.batch_size * self.module.max_seq_length,
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
                attack_grads, _ = self.module._parallel_forward(**kwargs)
                all_grads.append(attack_grads)
                grad, _ = utils.get_grad_and_param(
                    self.module.trainable_variables,
                    attack_grads, "word_embedding")
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
            attack_grads, _ = self.module._parallel_forward(**kwargs)
            all_grads.append(attack_grads)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [utils.average_n_grads(split_grad) for split_grad in zip(
                *all_grads)]
        update_params_op = utils.update_global_params(
            self.module.trainable_variables, self.module._global_step,
            self.module._optimizer, grads)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

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
                grads, tensors = self.module._parallel_forward(**kwargs)
                if k == 0:
                    self.module._tensors = tensors
                grad, param = utils.get_grad_and_param(
                    self.module.trainable_variables, grads, "word_embedding")
                update_params_op = utils.update_global_params(
                    self.module.trainable_variables, self.module._global_step,
                    self.module._optimizer, grads)

            # attack
            with tf.control_dependencies([update_params_op]):
                # any operator directly applied to `IndexedSlice` is dangerous
                sign = tf.cast(tf.greater(grad.values, 0.0), tf.float32)
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
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

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
        unused_grads, self.module._tensors = self.module._parallel_forward(**kwargs)
        cls_loss = tf.reduce_mean(self.module._tensors["losses"])

        # Bregman proximal point optimization
        param = utils.get_param(self.module.trainable_variables, "word_embedding")
        embedding_shape = param.shape.as_list()
        tilda = tf.get_variable(
            name="tilda_embeddings",
            shape=embedding_shape,
            initializer=tf.zeros_initializer, trainable=False)
        _, breg_tensors = self.module._parallel_forward(use_tilda_embedding=True, **kwargs)
        probs = self.module._tensors["probs"]
        probs_breg = breg_tensors["probs"]
        per_example_loss = tf.reduce_sum(
            probs_breg * (tf.log(probs_breg) - tf.log(probs)), axis=-1)
        per_example_loss_breg = tf.reduce_sum(
            probs * (tf.log(probs) - tf.log(probs_breg)), axis=-1)
        breg_loss = breg_miu * (
            tf.reduce_mean(per_example_loss) +
            tf.reduce_mean(per_example_loss_breg))
        self.module._tensors["breg"] = breg_miu * (
            per_example_loss +
            per_example_loss_breg)

        # perturbation
        grad, param = utils.get_grad_and_param(
            self.module.trainable_variables, unused_grads, "word_embedding")
        init_r = tf.get_variable(
            "init_r",
            shape=[self.module.batch_size * self.module.max_seq_length,
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
                _, prtb_tensors = self.module._parallel_forward(**kwargs)

                # smoothness-inducing adversarial regulization
                probs_prtb = prtb_tensors["probs"]
                per_example_loss = tf.reduce_sum(
                    probs_prtb * (tf.log(probs_prtb) - tf.log(probs)), axis=-1)
                per_example_loss_prtb = tf.reduce_sum(
                    probs * (tf.log(probs) - tf.log(probs_prtb)), axis=-1)
                prtb_loss = prtb_lambda * (
                    tf.reduce_mean(per_example_loss) +
                    tf.reduce_mean(per_example_loss_prtb))
                self.module._tensors["prtb"] = prtb_lambda * (
                    per_example_loss +
                    per_example_loss_prtb)

                # sum up
                total_loss = cls_loss + breg_loss + prtb_loss
                grads = tf.gradients(total_loss, self.module.trainable_variables)
                grad, _ = utils.get_grad_and_param(
                    self.module.trainable_variables, grads, "word_embedding")

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
            self.module.trainable_variables, self.module._global_step,
            self.module._optimizer, grads)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

        # runs at the start of traning
        self.init_tilda_op = tilda.assign(param)

        # runs at the end of each training epoch
        self.update_tilda_op = tilda.assign(
            (1 - tilda_beta) * param + tilda_beta * tilda)
