import os
import numpy as np
from abc import abstractmethod

from ..third import tf
from .. import com


class Task:
    """ Parent class of all tasks.

    This is an internal class that does not provide interface for outside requests."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def _init_session(self, ignore_checkpoint=False):
        """ Initialize Tensorflow session. """
        com.count_params(self.module.global_variables, self.module.trainable_variables)
        
        if self.module._gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.module._gpu_ids)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # disable GPUs
        config = tf.ConfigProto(allow_soft_placement=True)
        self.module.sess = tf.Session(graph=self.module.graph, config=config)
        self._init_variables(self.module.global_variables, ignore_checkpoint=ignore_checkpoint)
        self.module._session_built = True

    def _init_variables(self, variables, ignore_checkpoint=False):
        """ Initialize variables in the session. """

        # randomly initialize variables
        tf.logging.info("Running local_init_op")
        local_init_op = tf.variables_initializer(variables)
        self.module.sess.run(local_init_op)
        self.module._inited_vars |= set(variables)
        tf.logging.info("Done running local_init_op")

        # read from checkpoint file
        if not ignore_checkpoint and self.module.init_checkpoint:
            checkpoint_path = com.get_checkpoint_path(self.module.init_checkpoint)
            if not checkpoint_path:
                raise ValueError(
                    "Checkpoint file \"%s\" does not exist. Make sure you pass correct value to "
                    "`init_checkpoint`."
                    % self.module.init_checkpoint
                )
            self.module.init_checkpoint = checkpoint_path       # rectified path replacement

            # `continual` means we tend to succeed the training step and momentums variables "
            # "stored in the checkpoint file
            continual = os.path.dirname(checkpoint_path) == self.module.output_dir
            if continual:
                self.module.step = int(checkpoint_path.split("-")[-1])

            # build a bridge between the variables in checkpoint file and the variables in the graph
            (assignment_map, uninited_vars) = com.get_assignment_map(checkpoint_path, variables, continual=continual)
            self.module.assignment_map = assignment_map
            self.module.uninited_vars = uninited_vars

            if uninited_vars:
                tf.logging.info(
                    "%d local variables failed to match up with the checkpoint file. "
                    "Check more details through `.uninited_vars`." 
                    % len(uninited_vars)
                )

            if not self.module.assignment_map:    # no variables to restore
                return
            loader = tf.train.Saver(self.module.assignment_map)
            loader.restore(self.module.sess, checkpoint_path)

            if "_global_step" in self.module.__dict__:
                self.module.sess.run(tf.assign(self.module._global_step, self.module.step))

    def _build_feed_dict(self):
        """ Build `feed dict` for the current batch of data. """

        feed_dict = {}
        for key, data in self.module.data.items():
            if key.startswith(com.BACKUP_DATA):     # not to feed
                continue

            # move pointer and form the batch
            ptr = self._ptr
            batch = data[ptr: ptr + self.module.batch_size]
            ptr += self.module.batch_size

            # fill up the batch
            while len(batch) < self.module.batch_size:
                ptr = self.module.batch_size - len(batch)
                remainder = data[:ptr]
                concat_func = np.vstack if len(batch.shape) > 1 else np.hstack
                batch = concat_func((batch, remainder))

            placeholder = self.module.placeholders[key]
            feed_dict[placeholder] = batch

        self._ptr = ptr
        return feed_dict
