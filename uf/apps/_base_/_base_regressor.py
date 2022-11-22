import numpy as np

from ._base_ import BaseDecoder
from ...core import BaseModule
from ... import com
from ...third import tf
from .. import util


class RegDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        label_floats,
        label_size=2,
        sample_weight=None,
        scope="reg",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if kwargs.get("is_logits"):
            logits = input_tensor
        else:
            if kwargs.get("return_hidden"):
                self.tensors["hidden"] = input_tensor

            with tf.variable_scope(scope):
                output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
                intermediate_output = tf.layers.dense(
                    output_layer,
                    label_size * 4,
                    use_bias=False,
                    kernel_initializer=util.create_initializer(initializer_range),
                    trainable=trainable,
                )
                logits = tf.layers.dense(
                    intermediate_output,
                    label_size,
                    use_bias=False,
                    kernel_initializer=util.create_initializer(initializer_range),
                    trainable=trainable,
                    name="probs",
                )

        self.tensors["probs"] = logits

        per_example_loss = util.mean_squared_error(logits, label_floats, **kwargs)
        if sample_weight is not None:
            per_example_loss *= sample_weight
        self.tensors["losses"] = per_example_loss
        self.train_loss = tf.reduce_mean(per_example_loss)


class RegressorModule(BaseModule):
    """ Application class of regression. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def _convert_x(self, x, tokenized):
        """ Convert text sample. """

        # deal with untokenized inputs
        if not tokenized:

            # deal with general inputs
            if isinstance(x, str):
                return [self.tokenizer.tokenize(x)]

            # deal with multiple inputs
            return [self.tokenizer.tokenize(seg) for seg in x]

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return [x]

        # deal with tokenized and multiple inputs
        return x

    def _convert_y(self, y):

        sample = y[0]
        if isinstance(sample, list):
            self.label_size = len(sample)
        elif isinstance(sample, float) or isinstance(sample, int) or isinstance(sample, str):
            self.label_size = 1

        label_floats = []
        for idx, sample in enumerate(y):
            try:
                if isinstance(sample, list):
                    _label_floats = [float(label) for label in sample]
                elif isinstance(sample, float) or isinstance(sample, int) or isinstance(sample, str):
                    _label_floats = [float(sample)]
            except Exception as e:
                raise ValueError("Wrong label format (%s): %s. An example: y = [[0.12, 0.09], [-0.53, 0.98], ...]" % (sample, e))
            label_floats.append(_label_floats)

        return label_floats

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["probs"]]
        if from_tfrecords:
            ops.extend([self.placeholders["label_floats"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_labels = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_floats"]]

        # mse
        batch_preds = output_arrays[0]
        mse = np.mean(np.square(batch_preds - batch_labels))

        info = ""
        info += ", mse %.6f" % mse

        return info

    def _get_predict_ops(self):
        return [self.tensors["probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        outputs = {}
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self.tensors["probs"], self.tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # mse
        probs = com.transform(output_arrays[0], n_inputs)
        labels = self.data["label_floats"]
        mse = np.mean(np.square(probs - labels))

        outputs = {}
        outputs["mse"] = mse

        return outputs

