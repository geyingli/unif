import numpy as np

from ...core import BaseModule
from ... import com


class ClassifierModule(BaseModule):
    """ Application class of classification. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

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
        ops = [self._tensors["preds"], self._tensors["losses"]]
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
        return [self._tensors["probs"]]

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
        return [self._tensors["preds"], self._tensors["losses"]]

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
