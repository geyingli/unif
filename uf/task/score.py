import time

from ..third import tf
from .. import com
from .base import Task


class Scoring(Task):
    """ Infer the data and score the performance. """

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
            com.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session()
        self.module._session_mode = "infer"

        self._ptr = 0
        last_tic = time.time()
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.module.batch_size + 1
        for step in range(total_steps):
            self._score_one_batch(step, last_tic, total_steps, batch_outputs)

        return self.module._get_score_outputs(batch_outputs)

    def _score_one_batch(self, step, last_tic, total_steps, batch_outputs):
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
