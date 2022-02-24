import os
import time
import random
import numpy as np
import multiprocessing
from abc import abstractmethod

from ..tools import tf
from .. import utils


class Task:
    """ Parent class of all tasks.

    This is an internal class that does not provide interface
    for outside requests."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_session(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.module._gpu_ids)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.module.sess = tf.Session(graph=self.module.graph, config=config)
        self._init_variables(self.module.global_variables)
        self.module._session_built = True

    def _init_variables(self, variables, ignore_checkpoint=False):

        tf.logging.info("Running local_init_op")
        local_init_op = tf.variables_initializer(variables)
        self.module.sess.run(local_init_op)
        self.module._inited_vars |= set(variables)
        tf.logging.info("Done running local_init_op")

        if not ignore_checkpoint and self.module.init_checkpoint:
            checkpoint_path = utils.get_checkpoint_path(self.module.init_checkpoint)
            if not checkpoint_path:
                raise ValueError("Checkpoint file \"%s\" does not exist. "
                                 "Make sure you pass correct value to "
                                 "`init_checkpoint`."
                                 % self.module.init_checkpoint)
            self.module.init_checkpoint = checkpoint_path

            # `continual` means we tend to succeed the training step
            # and momentums variables in optimization
            continual = os.path.dirname(checkpoint_path) == self.module.output_dir
            if continual:
                self.module.step = int(checkpoint_path.split("-")[-1])

            (assignment_map, uninited_vars) = utils.get_assignment_map(
                checkpoint_path, variables, continual=continual)
            self.module.assignment_map = assignment_map
            self.module.uninited_vars = uninited_vars

            if uninited_vars:
                tf.logging.info(
                    "%d local variables failed to match up with the "
                    "checkpoint file. Check more details through "
                    "`.uninited_vars`." % len(uninited_vars))

            if not self.module.assignment_map:    # no variables to restore
                return
            loader = tf.train.Saver(self.module.assignment_map)
            loader.restore(self.module.sess, checkpoint_path)

            if "_global_step" in self.module.__dict__:
                self.module.sess.run(tf.assign(self.module._global_step, self.module.step))

    def _build_feed_dict(self):
        feed_dict = {}
        for key, data in self.module.data.items():
            if key.startswith(utils.BACKUP_DATA):
                continue
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


class Training(Task):

    def __init__(self, module, **kwargs):
        self.module = module
        self.grad_acc_steps = float(kwargs.get("grad_acc_steps", "1"))    # gradient accumulation
        self.shuffle = kwargs.get("shuffle", True)    # if to shuffle the training data
        self.adversarial = kwargs.get("adversarial", "").lower()    # adversarial training algorithm
        self.from_tfrecords = bool(kwargs.get("tfrecords_files"))    # if to reader data from tfrecords
        self.tfrecords_files = kwargs.get("tfrecords_files", [])    # paths of tfrecords
        self.n_jobs = kwargs.get("n_jobs", max(multiprocessing.cpu_count() - 1, 1))    # number of threads loading tfrecords
        self.max_to_keep = kwargs.get("max_to_keep", 1000000)

        self.decorate(**kwargs)

    def decorate(self, **kwargs):
        self._set_placeholders()

        # accumulate gradients for updation
        if self.grad_acc_steps > 1:
            self._accumulate_gradients(**kwargs)
        else:
            grads, self.module._tensors = self.module._parallel_forward(**kwargs)
            update_params_op = utils.update_global_params(
                self.module.trainable_variables, self.module._global_step, self.module._optimizer, grads)
            update_step_op = self.module._global_step.assign(self.module._global_step + 1)
            self.train_ops = [update_params_op, update_step_op]

    def _accumulate_gradients(self, **kwargs):
        grads, self.module._tensors = self.module._parallel_forward(**kwargs)

        params = []
        new_grads = []
        update_grad_ops = []
        for i, grad in enumerate(grads):
            if grad is None:
                continue

            param = self.module.trainable_variables[i]
            param_name = utils.get_param_name(param)

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
                    initializer=tf.zeros_initializer())

                # variable to store indices
                indices_shape = grad.indices.shape.as_list()    # [None]
                indices_shape[0] = int(n_elements * self.grad_acc_steps)
                indices_variable = tf.get_variable(
                    name=param_name + "/grad_indices",
                    shape=indices_shape,
                    dtype=tf.int32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

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
                    dense_shape=dense_shape)

                update_grad_op = tf.group([values_assign_op, indices_assign_op])
                new_grads.append(new_grad)
            else:
                grad_shape = grad.shape.as_list()
                grad_variable = tf.get_variable(
                    name=param_name + "/grad",
                    shape=grad_shape,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                new_grad = grad_variable + grad / self.grad_acc_steps
                update_grad_op = tf.assign(grad_variable, new_grad)
                new_grads.append(grad_variable)

            params.append(param)
            update_grad_ops.append(update_grad_op)

        update_grads_op = tf.group(update_grad_ops)
        update_step_op = self.module._global_step.assign(self.module._global_step + 1)
        self.train_ops = [update_grads_op, update_step_op]
        self.update_params_op = utils.update_global_params(
            params, self.module._global_step, self.module._optimizer, new_grads)

    def run(self, target_steps,
            print_per_secs=60,
            save_per_steps=1000):

        if self.shuffle and not self.tfrecords_files:
            self._shuffle()

        # init session
        if not self.module._session_built:
            utils.count_params(self.module.global_variables, self.module.trainable_variables)
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
                self.adversarial, self.n_inputs, self.module.step, target_steps)
        else:
            tf.logging.info(
                "Running training on %d samples (step %d -> %d)",
                self.n_inputs, self.module.step, target_steps)
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
                print_per_secs, save_per_steps, saver,
                self.adversarial)
            self.module.step += 1

            # use accumulated gradients to update parameters
            if (self.grad_acc_steps > 1) and (i % self.grad_acc_steps == 0):
                self.module.sess.run(self.update_params_op)

    def _set_placeholders(self):
        if self.from_tfrecords:
            self.n_inputs = utils.get_tfrecords_length(self.tfrecords_files)

            self.module._set_placeholders("feature", is_training=True)
            features = {key: self.module.placeholders[key]
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
            if tf.__version__.startswith("1"):
                map_and_batch = tf.contrib.data.map_and_batch
            elif tf.__version__.startswith("2"):
                map_and_batch = tf.data.experimental.map_and_batch
            dataset = dataset.apply(map_and_batch(
                decode_record,
                batch_size=self.module.batch_size,
                num_parallel_batches=self.n_jobs,
                drop_remainder=True))
            dataset = dataset.shuffle(buffer_size=100)
            iterator = dataset.make_one_shot_iterator()    # never stop
            self.module.placeholders = iterator.get_next()
        else:
            self.n_inputs = len(list(self.module.data.values())[0])
            self.module._set_placeholders("placeholder", is_training=True)

        if not self.n_inputs:
            raise ValueError("0 input samples recognized.")

    def _shuffle(self):
        index_list = list(range(len(list(self.module.data.values())[0])))
        random.shuffle(index_list)
        for key, data in self.module.data.items():
            if key.startswith(utils.BACKUP_DATA):
                continue
            self.module.data[key] = self.module.data[key][index_list]

    def _train_one_batch(self, step, last_tic, last_step, target_steps,
                         print_per_secs, save_per_steps, saver,
                         adversarial=None):
        feed_dict = {}
        as_feature = True
        if not self.from_tfrecords:
            feed_dict = self._build_feed_dict()
            as_feature = False
        fit_ops = self.module._get_fit_ops(as_feature) + self.train_ops

        output_arrays = self.module.sess.run(fit_ops, feed_dict=feed_dict)[:-len(self.train_ops)]

        # print
        if time.time() - last_tic > print_per_secs or step == target_steps:
            info = "step %d" % step

            # print processor-specific information
            info += self.module._get_fit_info(output_arrays, feed_dict, as_feature)

            # print training efficiency
            if time.time() - last_tic > print_per_secs or step == target_steps:
                info += ", %.2f steps/sec" % ((step - last_step) / (time.time() - last_tic))
                info += ", %.2f examples/sec" % ((step - last_step) / (time.time() - last_tic) * self.module.batch_size)

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


class Inference(Task):

    def __init__(self, module):
        self.module = module

        # ignore redundant building of the work flow
        if self.module._session_mode != "infer":
            self.decorate()

    def decorate(self):
        self.module._set_placeholders("placeholder", is_training=False)

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self):
        n_inputs = len(list(self.module.data.values())[0])
        if not n_inputs:
            raise ValueError("0 input samples recognized.")

        # init session
        if not self.module._session_built:
            utils.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session()
        self.module._session_mode = "infer"

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.module.batch_size + 1
        for step in range(total_steps):
            self._predict_one_batch(step, last_tic, total_steps, batch_outputs)

        return self.module._get_predict_outputs(batch_outputs)

    def _predict_one_batch(self, step, last_tic,
                           total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        predict_ops = self.module._get_predict_ops()
        output_arrays = self.module.sess.run(predict_ops, feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = "Time usage %dm-%.2fs" % (diff_tic // 60, diff_tic % 60)
            info += ", %.2f steps/sec" % (total_steps / diff_tic)
            info += ", %.2f examples/sec" % (total_steps / diff_tic * self.module.batch_size)

            tf.logging.info(info)


class Scoring(Task):

    def __init__(self, module):
        self.module = module

        # ignore redundant building of the work flow
        if self.module._session_mode != "infer":
            self.decorate()

    def decorate(self):
        self.module._set_placeholders("placeholder", is_training=False)

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self):
        n_inputs = len(list(self.module.data.values())[0])
        if not n_inputs:
            raise ValueError("0 input samples recognized.")

        # init session
        if not self.module._session_built:
            utils.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session()
        self.module._session_mode = "infer"

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.module.batch_size + 1
        for step in range(total_steps):
            self._score_one_batch(step, last_tic, total_steps, batch_outputs)

        return self.module._get_score_outputs(batch_outputs)

    def _score_one_batch(self, step, last_tic,
                         total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        score_ops = self.module._get_score_ops()
        output_arrays = self.module.sess.run(score_ops, feed_dict=feed_dict)

        # cache
        batch_outputs.append(output_arrays)

        # print
        if step == total_steps - 1:

            # print inference efficiency
            diff_tic = time.time() - last_tic
            info = "Time usage %dm-%.2fs" % (diff_tic // 60, diff_tic % 60)
            info += ", %.2f steps/sec" % (total_steps / diff_tic)
            info += ", %.2f examples/sec" % (total_steps / diff_tic * self.module.batch_size)

            tf.logging.info(info)


class Initialization(Task):

    def __init__(self, module):
        self.module = module

        self.decorate()

    def decorate(self):
        self.module._set_placeholders("placeholder", is_training=False)

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self, reinit_all, ignore_checkpoint):

        # init session
        if not self.module._session_built:
            utils.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session(ignore_checkpoint)
        elif reinit_all:
            self._init_session(ignore_checkpoint)
        else:
            variables = []
            for var in self.module.global_variables:
                if var not in self.module._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables, ignore_checkpoint)
            else:
                tf.logging.info(
                    "Global variables already initialized. "
                    "To re-initialize all, pass `reinit_all` to True.")
        self.module._session_mode = "infer"

    def _init_session(self, ignore_checkpoint):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.module._gpu_ids)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.module.sess = tf.Session(graph=self.module.graph, config=config)
        self._init_variables(self.module.global_variables, ignore_checkpoint)
        self.module._session_built = True


class Exportation(Task):

    def __init__(self, module):
        self.module = module

        self.decorate()

    def decorate(self):
        self.module._set_placeholders("placeholder", on_export=True, is_training=False)

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self, export_dir, rename_inputs=None, rename_outputs=None,
            ignore_inputs=None, ignore_outputs=None):

        # init session
        if not self.module._session_built:
            utils.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session()
        self.module._session_mode = "infer"

        def set_input(key, value):
            inputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info("Register Input: %s, %s, %s" % (
                key, value.shape.as_list(), value.dtype.name))

        # define inputs
        inputs = {}
        if not ignore_inputs:
            ignore_inputs = []
        for key, value in list(self.module.placeholders.items()):
            if key == "sample_weight" or key in ignore_inputs:
                continue
            if rename_inputs and key in rename_inputs:
                key = rename_inputs[key]
            set_input(key, value)

        def set_output(key, value):
            outputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info("Register Output: %s, %s, %s" % (
                key, value.shape.as_list(), value.dtype.name))

        # define outputs
        outputs = {}
        if not ignore_outputs:
            ignore_outputs = []
        for key, value in self.module._tensors.items():
            if key in ignore_outputs:
                continue
            if rename_outputs and key in rename_outputs:
                key = rename_outputs[key]
            set_output(key, value)

        # build signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs,
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        signature_def_map = {"predict": signature}
        tf.logging.info("Register Signature: predict")

        legacy_init_op = tf.group(
            tf.tables_initializer(), name="legacy_init_op")
        builder = tf.saved_model.builder.SavedModelBuilder(
            os.path.join(export_dir, time.strftime("%Y%m%d.%H%M%S")))
        try:
            builder.add_meta_graph_and_variables(
                self.module.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                legacy_init_op=legacy_init_op)
        except Exception:
            raise ValueError(
                "Twice exportation is not allowed. Try `.save()` and "
                "`.reset()` method to save and reset the graph before "
                "next exportation.")
        builder.save()
