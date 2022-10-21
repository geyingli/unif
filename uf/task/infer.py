import time

from ..third import tf
from ._base_ import Task


class Inference(Task):
    """ Inference, as its name. """

    def run(self):
        n_inputs = len(list(self.module.data.values())[0])
        if not n_inputs:
            raise ValueError("0 input samples recognized.")

        # build graph
        if self.module._session_mode != "infer" and not self.module._debug:
            self._build_graph()

        # init session
        if not self.module._session_built:
            self._init_session()
        self.module._session_mode = "infer"

        tf.logging.info("Running inference on %d samples", n_inputs)

        # inference
        self._ptr = 0
        last_tic = time.time()
        last_step = 0
        batch_outputs = []
        total_steps = (n_inputs - 1) // self.module.batch_size + 1
        for step in range(total_steps):
            last_tic, last_step = self._predict_one_batch(
                step + 1, last_tic, last_step, total_steps, batch_outputs,
            )

        output_arrays = list(zip(*batch_outputs))
        return self.module._get_predict_outputs(output_arrays, n_inputs)

    def _predict_one_batch(self, step, last_tic, last_step, total_steps, batch_outputs):
        feed_dict = self._build_feed_dict()
        predict_ops = self.module._get_predict_ops()
        output_arrays = self.module.sess.run(predict_ops, feed_dict=feed_dict)
        batch_outputs.append(output_arrays)

        # print
        diff_tic = time.time() - last_tic
        process = step / total_steps
        if (diff_tic > 10 and process >= 0.005) or step == total_steps:
            info = "process %.1f%%" % (process * 100)

            # print inference efficiency
            info += ", %.2f examples/sec" % ((step - last_step) / diff_tic * self.module.batch_size)

            tf.logging.info(info)
            last_tic = time.time()
            last_step = step

        return last_tic, last_step
