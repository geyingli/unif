import time

from ..third import tf
from ._base_ import Task


class Scoring(Task):
    """ Infer the data and score the performance. """

    def __init__(self, module):
        self.module = module

        # ignore redundant building of the work flow
        if self.module._session_mode != "infer":
            self.decorate()

    def decorate(self):
        self.module._set_placeholders()

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self):
        n_inputs = len(list(self.module.data.values())[0])
        if not n_inputs:
            raise ValueError("0 input samples recognized.")

        # init session
        if not self.module._session_built:
            self._init_session()
        self.module._session_mode = "infer"

        tf.logging.info("Running scoring on %d samples", n_inputs)

        self._ptr = 0
        last_tic = time.time()
        last_step = 0
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.module.batch_size + 1
        for step in range(total_steps):
            last_tic, last_step = self._score_one_batch(
                step + 1, last_tic, last_step, total_steps, batch_outputs,
            )

        output_arrays = list(zip(*batch_outputs))
        return self.module._get_score_outputs(output_arrays, n_inputs)

    def _score_one_batch(self, step, last_tic, last_step, total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        score_ops = self.module._get_score_ops()
        output_arrays = self.module.sess.run(score_ops, feed_dict=feed_dict)
        batch_outputs.append(output_arrays)

        # print
        diff_tic = time.time() - last_tic
        process = step / total_steps
        if (diff_tic > 10 and process >= 0.005) or step == total_steps:
            info = "process %.1f%%" % (process * 100)

            # print scoring efficiency
            info += ", %.2f examples/sec" % ((step - last_step) / diff_tic * self.module.batch_size)

            tf.logging.info(info)
            last_tic = time.time()
            last_step = step

        return last_tic, last_step
