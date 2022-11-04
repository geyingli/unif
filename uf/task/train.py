import time
import random
import multiprocessing

from ..third import tf
from .. import com
from ._base_ import Task


class Training(Task):
    """ Simply train, in a common neural-network manner. """

    def run(self, target_steps, print_per_secs=60, save_per_steps=1000, **kwargs):
        self.grad_acc_steps = float(kwargs.get("grad_acc_steps", "1"))      # gradient accumulation
        self.shuffle = kwargs.get("shuffle", True)                          # if to shuffle the training data
        self.adversarial = kwargs.get("adversarial", "").lower()            # adversarial training algorithm
        self.from_tfrecords = bool(kwargs.get("tfrecords_files"))           # if to reader data from tfrecords
        self.tfrecords_files = kwargs.get("tfrecords_files", [])            # paths of tfrecords
        self.n_jobs = kwargs.get("n_jobs", max(multiprocessing.cpu_count() - 1, 1))    # number of threads loading tfrecords
        self.max_to_keep = kwargs.get("max_to_keep", 1000000)

        # confirm inputs
        if self.from_tfrecords:
            self.n_inputs = com.get_tfrecords_length(self.tfrecords_files)
        else:
            self.n_inputs = len(list(self.module.data.values())[0])
        if not self.n_inputs:
            raise ValueError("0 input samples recognized.")

        # build graph
        if self.module._graph_mode != "train" or not self._debug:
            self._build_graph(**kwargs)

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

        # shuffle training samples
        if self.shuffle and not self.tfrecords_files:
            self._shuffle()

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

    def _convert_placeholders(self):
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

    def _build_graph(self, **kwargs):
        self.module._graph_mode = "train"
        self.module._set_placeholders(is_training=True)

        # convert placeholders into TFRecords-features
        if self.from_tfrecords:
            self._convert_placeholders()

        # accumulate gradients for updation
        if self.grad_acc_steps > 1:
            self._accumulate_gradients(**kwargs)
        else:
            grads, self.module.tensors = self.module._parallel_forward(**kwargs)
            update_params_op = com.update_global_params(
                self.module.trainable_variables,
                self.module._global_step,
                self.module._optimizer,
                grads,
            )
            update_step_op = self.module._global_step.assign(self.module._global_step + 1)
            self.train_ops = [update_params_op, update_step_op]

    def _accumulate_gradients(self, **kwargs):
        # Model tends to be harder to converge.
        #
        # Here remains problems to be discovered and solved.
        # For now, don't use gradient accumulation.

        grads, self.module.tensors = self.module._parallel_forward(**kwargs)

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
