import numpy as np

from ._base_ import BaseDecoder
from ...core import BaseModule
from ...third import tf
from ... import com
from .. import util


class ClsDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        label_ids,
        label_size=2,
        sample_weight=None,
        scope="cls",
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

        self.tensors["preds"] = tf.argmax(logits, axis=-1, name="preds")
        self.tensors["probs"] = tf.nn.softmax(logits, axis=-1, name="probs")

        per_example_loss = util.cross_entropy(logits, label_ids, label_size, **kwargs)
        if sample_weight is not None:
            per_example_loss *= sample_weight
        self.tensors["losses"] = per_example_loss
        self.train_loss = tf.reduce_mean(per_example_loss)


class ClassifierModule(BaseModule):
    """ Application class of classification. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def _get_f1(preds, labels, label_index=1):
        """ F1. """
        assert len(preds) == len(labels)
        preds = np.array(preds)
        labels = np.array(labels)

        tp, fp, fn, tn = 0, 0, 0, 0
        for pred, label in zip(preds, labels):
            if pred == label == label_index:
                tp += 1
            elif pred == label_index and label != label_index:
                fp += 1
            elif pred != label_index and label == label_index:
                fn += 1
            else:
                tn += 1
        n = tp + fn

        tp = np.sum(labels == label_index)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)
        return (n, accuracy, precision, recall, f1)

    def _get_best_f1(probs, labels, label_index=1):
        """ Best F1 with a certain threshold. """
        assert len(probs) == len(labels)
        probs = np.array(probs)
        labels = np.array(labels)

        # initialize metrics
        n = np.sum(labels == label_index)
        tp = n
        fp = len(labels) - n
        fn = 0
        tn = 0
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)
        threshold = 0

        ids = sorted(range(len(probs)), key=lambda i: probs[i])
        for i in ids:
            prob = probs[i]
            label = labels[i]
            if label == label_index:
                tp -= 1
                fn += 1
            elif label != label_index:
                fp -= 1
                tn += 1

            _accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
            _precision = tp / max(tp + fp, 1)
            _recall = tp / max(tp + fn, 1)
            _f1 = 2 * _precision * _recall / max(_precision + _recall, 1)
            if _f1 > f1:
                accuracy = _accuracy
                precision = _precision
                recall = _recall
                f1 = _f1
                threshold = prob
        return (n, accuracy, precision, recall, f1, threshold)

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
        """ Convert categorical label. """

        label_set = set(y)

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, "Number of unique `y`s exceeds `label_size`."
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
        for label in y:
            if label not in self._label_to_id:
                assert len(self._label_to_id) < self.label_size, "Number of unique labels exceeds `label_size`."
                self._label_to_id[label] = len(self._label_to_id)
                self._id_to_label.append(label)
            label_ids.append(self._label_to_id[label])
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
        preds = np.argmax(probs, axis=-1).tolist()
        if self._id_to_label:
            preds = [self._id_to_label[idx] if idx < len(self._id_to_label) else None for idx in preds]

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
