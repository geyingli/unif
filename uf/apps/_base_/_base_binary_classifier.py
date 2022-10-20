import numpy as np

from ._base_ import BaseDecoder
from ...core import BaseModule
from ...third import tf
from ... import com
from .. import util


class BinaryClsDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        label_ids,
        is_logits=False,
        label_size=2,
        sample_weight=None,
        label_weight=None,
        scope="cls/seq_relationship",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if is_logits:
            logits = input_tensor
        else:
            hidden_size = input_tensor.shape.as_list()[-1]
            with tf.variable_scope(scope):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[label_size, hidden_size],
                    initializer=util.create_initializer(initializer_range),
                    trainable=trainable,
                )
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[label_size],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable,
                )
                output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)

        probs = tf.nn.sigmoid(logits, name="probs")
        self.tensors["probs"] = probs
        self.tensors["preds"] = tf.greater(probs, 0.5, name="preds")

        per_label_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(label_ids, dtype=tf.float32))
        if label_weight is not None:
            label_weight = tf.constant(label_weight, dtype=tf.float32)
            label_weight = tf.reshape(label_weight, [1, label_size])
            per_label_loss *= label_weight
        per_example_loss = tf.reduce_sum(per_label_loss, axis=-1)
        if sample_weight is not None:
            per_example_loss *= sample_weight

        self.tensors["losses"] = per_example_loss
        self.train_loss = tf.reduce_mean(per_example_loss)


class BinaryClassifierModule(BaseModule):
    """ Application class of classification. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
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
        try:
            label_set = set()
            for sample in y:
                _label_set = set()
                for _y in sample:
                    assert _y not in _label_set
                    label_set.add(_y)
                    _label_set.add(_y)
        except Exception:
            raise ValueError("The element of `y` should be a list of multiple answers. E.g. y=[[1, 3], [0], [0, 2]].")

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, "Number of unique labels exceeds `label_size`."
        else:
            self.label_size = len(label_set)

        # automatically set `id_to_label`
        if not self._id_to_label:
            self._id_to_label = list(label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._id_to_label = list(sorted(self._id_to_label))
            except Exception:
                pass

        # automatically set `label_to_id` for prediction
        if not self._label_to_id:
            self._label_to_id = {label: index for index, label in enumerate(self._id_to_label)}

        label_ids = []
        for sample in y:
            _label_ids = [0] * self.label_size

            for label in sample:
                if label not in self._label_to_id:
                    assert len(self._label_to_id) < self.label_size, "Number of unique labels exceeds `label_size`."
                    self._label_to_id[label] = len(self._label_to_id)
                    self._id_to_label.append(label)
                label_id = self._label_to_id[label]
                _label_ids[label_id] = 1
            label_ids.append(_label_ids)
        return label_ids

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_labels = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # accuracy
        batch_preds = output_arrays[0]
        accuracy = np.mean(batch_preds == batch_labels)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", accuracy %.4f" % accuracy
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        preds = (probs >= 0.5)
        if self._id_to_label:
            preds = [[
                self._id_to_label[i] for i in range(self.label_size)
                if _preds[i] and i < len(self._id_to_label)
            ] for _preds in preds]
        else:
            preds = [[i for i in range(self.label_size) if _preds[i]] for _preds in preds]

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self.tensors["preds"], self.tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # accuracy
        preds = com.transform(output_arrays[0], n_inputs)
        labels = self.data["label_ids"]
        accuracy = np.mean(preds == labels)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["accuracy"] = accuracy
        outputs["loss"] = loss

        return outputs


