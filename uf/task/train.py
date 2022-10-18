import time
import random
import numpy as np
import multiprocessing

from ..third import tf
from .. import com
from ._base_ import Task


class Training(Task):
    """ Simply train, in a common neural-network manner. """

    def __init__(self, module, **kwargs):
        self.module = module
        self.grad_acc_steps = float(kwargs.get("grad_acc_steps", "1"))      # gradient accumulation
        self.shuffle = kwargs.get("shuffle", True)                          # if to shuffle the training data
        self.adversarial = kwargs.get("adversarial", "").lower()            # adversarial training algorithm
        self.from_tfrecords = bool(kwargs.get("tfrecords_files"))           # if to reader data from tfrecords
        self.tfrecords_files = kwargs.get("tfrecords_files", [])            # paths of tfrecords
        self.n_jobs = kwargs.get("n_jobs", max(multiprocessing.cpu_count() - 1, 1))    # number of threads loading tfrecords
        self.max_to_keep = kwargs.get("max_to_keep", 1000000)

        self.decorate(**kwargs)

    def decorate(self, **kwargs):
        self._init_placeholders()

        # accumulate gradients for updation
        if self.grad_acc_steps > 1:
            self._accumulate_gradients(**kwargs)
        else:
            grads, self.module._tensors = self.module._parallel_forward(**kwargs)
            update_params_op = com.update_global_params(
                self.module.trainable_variables,
                self.module._global_step,
                self.module._optimizer,
                grads,
            )
            update_step_op = self.module._global_step.assign(self.module._global_step + 1)
            self.train_ops = [update_params_op, update_step_op]

    def run(self, target_steps, print_per_secs=60, save_per_steps=1000):

        # shuffle training samples
        if self.shuffle and not self.tfrecords_files:
            self._shuffle()

        # init session/variables
        if not self.module._session_built:
            self._init_session()
        else:
            variables = []
            for var in self.module.global_variables:
                if var not in self.module._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables)
        self.module._session_mode = "train"

        # print
        if self.adversarial:
            tf.logging.info(
                "Running adversarial training `%s` on %d samples (step %d -> %d)",
                self.adversarial, self.n_inputs, self.module.step, target_steps,
            )
        else:
            tf.logging.info(
                "Running training on %d samples (step %d -> %d)",
                self.n_inputs, self.module.step, target_steps,
            )
        if self.grad_acc_steps > 1:
            tf.logging.info("Accumulate gradients every %d steps" % self.grad_acc_steps)

        # SMART: initialize tilda_embedding
        if self.adversarial == "smart":
            self.module.sess.run(self.init_tilda_op)

        self._ptr = 0
        last_tic = time.time()
        last_step = self.module.step
        saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        for i in range(target_steps - self.module.step):
            last_tic, last_step = self._train_one_batch(
                self.module.step + 1, last_tic, last_step, target_steps,
                print_per_secs, save_per_steps, saver, self.adversarial,
            )
            self.module.step += 1

            # use accumulated gradients to update parameters
            if (self.grad_acc_steps > 1) and (i % self.grad_acc_steps == 0):
                self.module.sess.run(self.update_params_op)

    def _init_placeholders(self):
        if self.from_tfrecords:
            self.n_inputs = com.get_tfrecords_length(self.tfrecords_files)

            self.module._set_placeholders(is_training=True)

            # convert placeholders into features
            features = {}
            for key in com.get_tfrecords_keys(self.tfrecords_files[0]):
                feature = self.module.placeholders[key]
                if not isinstance(feature, tf.FixedLenFeature):
                    feature = com.convert_placeholder_to_feature(feature)
                features[key] = feature

            def decode_record(record):
                example = tf.parse_single_example(record, features)
                for name in list(example.keys()):
                    _t = example[name]
                    if _t.dtype == tf.int64:
                        _t = tf.to_int32(_t)
                    example[name] = _t
                return example

            dataset = tf.data.TFRecordDataset(self.tfrecords_files)
            dataset = dataset.repeat()
            if tf.__version__.startswith("1"):
                map_and_batch = tf.contrib.data.map_and_batch
            elif tf.__version__.startswith("2"):
                map_and_batch = tf.data.experimental.map_and_batch
            dataset = dataset.apply(map_and_batch(
                decode_record,
                batch_size=self.module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True),
            )
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            self.module.placeholders = iterator.get_next()
        else:
            self.n_inputs = len(list(self.module.data.values())[0])
            self.module._set_placeholders(is_training=True)

        if not self.n_inputs:
            raise ValueError("0 input samples recognized.")

    def _accumulate_gradients(self, **kwargs):
        grads, self.module._tensors = self.module._parallel_forward(**kwargs)

        params = []
        new_grads = []
        update_grad_ops = []
        for i, grad in enumerate(grads):
            if grad is None:
                continue

            param = self.module.trainable_variables[i]
            param_name = com.get_param_name(param)

            if grad.__str__().startswith("IndexedSlices"):
                dense_shape = grad.dense_shape
                n_elements = self.module.batch_size * self.module.max_seq_length

                # variable to store values
                values_shape = grad.values.shape.as_list()    # [None, max_seq_length]
                values_shape[0] = int(n_elements * self.grad_acc_steps)
                values_variable = tf.get_variable(
                    name=param_name + "/grad_values",
                    shape=values_shape,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer(),
                )

                # variable to store indices
                indices_shape = grad.indices.shape.as_list()    # [None]
                indices_shape[0] = int(n_elements * self.grad_acc_steps)
                indices_variable = tf.get_variable(
                    name=param_name + "/grad_indices",
                    shape=indices_shape,
                    dtype=tf.int32,
                    trainable=False,
                    initializer=tf.zeros_initializer(),
                )

                # add new values
                new_values = tf.concat([values_variable[n_elements:], grad.values / self.grad_acc_steps], axis=0)
                values_assign_op = tf.assign(values_variable, new_values)

                # add new indices
                new_indices = tf.concat([indices_variable[n_elements:], grad.indices], axis=0)
                indices_assign_op = tf.assign(indices_variable, new_indices)

                # obtain new gradient
                new_grad = tf.IndexedSlices(
                    values=values_variable,
                    indices=indices_variable,
                    dense_shape=dense_shape,
                )

                update_grad_op = tf.group([values_assign_op, indices_assign_op])
                new_grads.append(new_grad)
            else:
                grad_shape = grad.shape.as_list()
                grad_variable = tf.get_variable(
                    name=param_name + "/grad",
                    shape=grad_shape,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer(),
                )
                new_grad = grad_variable + grad / self.grad_acc_steps
                update_grad_op = tf.assign(grad_variable, new_grad)
                new_grads.append(grad_variable)

            params.append(param)
            update_grad_ops.append(update_grad_op)

        update_grads_op = tf.group(update_grad_ops)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_grads_op, update_step_op]
        self.update_params_op = com.update_global_params(
            params,
            self.module._global_step,
            self.module._optimizer,
            new_grads,
        )

    def _shuffle(self):
        index_list = list(range(len(list(self.module.data.values())[0])))
        random.shuffle(index_list)

        # indexing
        shuffled_data = {}
        for key in self.module.data.keys():
            if key.startswith(com.BACKUP_DATA):
                continue
            
            # Error may occurs when the system trying to unberably 
            # allocate large memory for indexing. 
            try:
                shuffled_data[key] = self.module.data[key][index_list]
            except:
                return

        # replace
        for key in shuffled_data:
            self.module.data[key] = shuffled_data[key]

    def _train_one_batch(self, step, last_tic, last_step, target_steps, print_per_secs, save_per_steps, saver, adversarial=None):
        feed_dict = {}
        if not self.from_tfrecords:
            feed_dict = self._build_feed_dict()
        fit_ops = self.module._get_fit_ops(self.from_tfrecords) + self.train_ops

        output_arrays = self.module.sess.run(fit_ops, feed_dict=feed_dict)[:-len(self.train_ops)]

        # print
        diff_tic = time.time() - last_tic
        if diff_tic > print_per_secs or step == target_steps:
            info = "step %d" % step

            # print processor-specific information
            info += self.module._get_fit_info(output_arrays, feed_dict, self.from_tfrecords)

            # print training efficiency
            info += ", %.2f steps/sec" % ((step - last_step) / diff_tic)
            info += ", %.2f examples/sec" % ((step - last_step) / diff_tic * self.module.batch_size)

            tf.logging.info(info)
            last_tic = time.time()
            last_step = step

        # SMART: update tilda_embedding
        if step % self.module.steps_per_epoch == 0 and adversarial == "smart":
            self.module.sess.run(self.update_tilda_op)

        # save
        if self.module.output_dir and step % save_per_steps == 0:
            tf.logging.info("Saving checkpoint for %d into %s/model.ckpt" % (step, self.module.output_dir))
            self.module.init_checkpoint = (self.module.output_dir + "/model.ckpt-%d" % step)
            saver.save(self.module.sess, self.module.init_checkpoint)

        return last_tic, last_step


class AdversarialTraining(Training):
    """ Train with adversarial algorithms. """

    def decorate(self, **kwargs):
        self._init_placeholders()

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
            raise ValueError(
                "%s does not support adversarial algorithm `%s`." 
                % (self.module.__class__.__name__, self.adversarial)
            )
        if not ok:
            raise ValueError(
                "Wrong adversarial algorithm `%s`. Pick one in the following list: "
                "FGM, PGD, FreeLB, FreeAT, SMART." % self.adversarial
            )

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
        grad, param = com.get_grad_and_param(self.module.trainable_variables, actual_grads, "word_embedding")
        r = tf.multiply(epsilon, grad / (tf.norm(grad) + 1e-9))
        attack_op = param.assign(param + r)

        # restore
        with tf.control_dependencies([attack_op]):
            attack_grads, _ = self.module._parallel_forward(**kwargs)
            restore_op = param.assign(param - r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [
                com.average_n_grads([actual_grad, attack_grad])
                for (actual_grad, attack_grad) in zip(actual_grads, attack_grads)
            ]
        update_params_op = com.update_global_params(
            self.module.trainable_variables,
            self.module._global_step,
            self.module._optimizer,
            grads,
        )
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
                grad, param = com.get_grad_and_param(self.module.trainable_variables, d_grads, "word_embedding")
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = tf.cond(
                    norm > epsilon,
                    lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                    lambda: (acc_r + tmp_r),
                )
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # restore
        with tf.control_dependencies([attack_op]):
            attack_grads, _ = self.module._parallel_forward(**kwargs)
            restore_op = param.assign(param - acc_r)

        # sum up
        with tf.control_dependencies([restore_op]):
            grads = [
                com.average_n_grads([actual_grad, attack_grad])
                for (actual_grad, attack_grad) in zip(actual_grads, attack_grads)
            ]
        update_params_op = com.update_global_params(
            self.module.trainable_variables,
            self.module._global_step,
            self.module._optimizer,
            grads,
        )
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
        grad, param = com.get_grad_and_param(self.module.trainable_variables, d_grads, "word_embedding")
        init_r = tf.get_variable(
            "init_r",
            shape=[self.module.batch_size * self.module.max_seq_length, param.shape.as_list()[-1]],
            initializer=tf.random_uniform_initializer(minval=-epsilon, maxval=epsilon),
            trainable=False,
        )
        init_op = tf.variables_initializer([init_r])
        with tf.control_dependencies([init_op]):    # fix perturbation
            # Scale randomly initialized permutation, to make sure norm
            # of `r` is smaller than epsilon.
            r = tf.divide(init_r, tf.norm(init_r, np.inf))
            r = tf.IndexedSlices(values=r, indices=grad.indices, dense_shape=grad.dense_shape)
            attack_op = param.assign(param + r)

        # attack
        acc_r = r
        all_grads = []
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                attack_grads, _ = self.module._parallel_forward(**kwargs)
                all_grads.append(attack_grads)
                grad, _ = com.get_grad_and_param(self.module.trainable_variables, attack_grads, "word_embedding")
                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad) + 1e-9))

                # In order not to shuffle the distribution of gradient-
                # induced perturbation, we use norm to scale instead of
                # simply clip the values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = tf.cond(
                    norm > epsilon,
                    lambda: (acc_r + tmp_r) * tf.divide(epsilon, norm),
                    lambda: (acc_r + tmp_r),
                )
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
            grads = [com.average_n_grads(split_grad) for split_grad in zip(*all_grads)]
        update_params_op = com.update_global_params(
            self.module.trainable_variables,
            self.module._global_step,
            self.module._optimizer,
            grads,
        )
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
                grad, param = com.get_grad_and_param(self.module.trainable_variables, grads, "word_embedding")
                update_params_op = com.update_global_params(
                    self.module.trainable_variables,
                    self.module._global_step,
                    self.module._optimizer,
                    grads,
                )

            # attack
            with tf.control_dependencies([update_params_op]):
                # any operator directly applied to `IndexedSlice` is dangerous
                sign = tf.cast(tf.greater(grad.values, 0.0), tf.float32)
                r = last_r + tf.multiply(epsilon, sign) if k > 0 else tf.multiply(epsilon, sign)
                r = tf.cond(
                    tf.norm(r) > epsilon,
                    lambda: r * tf.divide(epsilon, tf.norm(r)),
                    lambda: r,
                )
                r_slice = tf.IndexedSlices(values=r, indices=grad.indices, dense_shape=grad.dense_shape)
                attack_op = param.assign(param - last_r_slice + r_slice)
                last_r = r
                last_r_slice = r_slice
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

    def _smart(self, epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3, **kwargs):
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
        param = com.get_param(self.module.trainable_variables, "word_embedding")
        embedding_shape = param.shape.as_list()
        tilda_embeddings = tf.get_variable(
            name="tilda_embeddings",
            shape=embedding_shape,
            initializer=tf.zeros_initializer,
            trainable=False,
        )
        _, breg_tensors = self.module._parallel_forward(tilda_embeddings=tilda_embeddings, **kwargs)
        probs = self.module._tensors["probs"]
        probs_breg = breg_tensors["probs"]
        per_example_loss = tf.reduce_sum(probs_breg * (tf.log(probs_breg) - tf.log(probs)), axis=-1)
        per_example_loss_breg = tf.reduce_sum(probs * (tf.log(probs) - tf.log(probs_breg)), axis=-1)
        breg_loss = breg_miu * (tf.reduce_mean(per_example_loss) + tf.reduce_mean(per_example_loss_breg))
        self.module._tensors["breg"] = breg_miu * (per_example_loss + per_example_loss_breg)

        # perturbation
        grad, param = com.get_grad_and_param(self.module.trainable_variables, unused_grads, "word_embedding")
        init_r = tf.get_variable(
            "init_r",
            shape=[self.module.batch_size * self.module.max_seq_length, embedding_shape[-1]],
            initializer=tf.random_normal_initializer(stddev=epsilon),
            trainable=False,
        )
        with tf.control_dependencies([breg_loss]):
            init_op = tf.variables_initializer([init_r])
        with tf.control_dependencies([init_op]):    # fix perturbation
            # Scale randomly initialized permutation, to make sure norm
            # of `r` is smaller than epsilon.
            r = tf.divide(init_r, tf.norm(init_r, np.inf))
            r = tf.IndexedSlices(values=r, indices=grad.indices, dense_shape=grad.dense_shape)
            attack_op = param.assign(param + r)

        # attack
        acc_r = r
        for k in range(n_loop):
            with tf.control_dependencies([attack_op]):
                _, prtb_tensors = self.module._parallel_forward(**kwargs)

                # smoothness-inducing adversarial regulization
                probs_prtb = prtb_tensors["probs"]
                per_example_loss = tf.reduce_sum(probs_prtb * (tf.log(probs_prtb) - tf.log(probs)), axis=-1)
                per_example_loss_prtb = tf.reduce_sum(probs * (tf.log(probs) - tf.log(probs_prtb)), axis=-1)
                prtb_loss = prtb_lambda * (tf.reduce_mean(per_example_loss) + tf.reduce_mean(per_example_loss_prtb))
                self.module._tensors["prtb"] = prtb_lambda * (per_example_loss + per_example_loss_prtb)

                # sum up
                train_loss = cls_loss + breg_loss + prtb_loss
                grads = tf.gradients(train_loss, self.module.trainable_variables)
                grad, _ = com.get_grad_and_param(self.module.trainable_variables, grads, "word_embedding")

                tmp_r = tf.multiply(1 / n_loop, grad / (tf.norm(grad, np.inf) + 1e-9))

                # In order not to shuffle the distribution of gradient-induced
                # perturbation, we use norm to scale instead of simply clip the
                # values.
                norm = tf.norm(acc_r + tmp_r)
                cur_r = (acc_r + tmp_r) * tf.divide(epsilon, norm)
                r = cur_r - acc_r    # calculate current step
                attack_op = param.assign(param + r)
                acc_r = cur_r

        # update
        update_params_op = com.update_global_params(
            self.module.trainable_variables,
            self.module._global_step,
            self.module._optimizer,
            grads,
        )
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_params_op, update_step_op]

        # runs at the start of traning
        self.init_tilda_op = tilda_embeddings.assign(param)

        # runs at the end of each training epoch
        self.update_tilda_op = tilda_embeddings.assign((1 - tilda_beta) * param + tilda_beta * tilda_embeddings)
