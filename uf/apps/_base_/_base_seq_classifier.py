import numpy as np

from ._base_ import BaseDecoder
from ...core import BaseModule
from ...third import tf
from ... import com
from .. import util


class SeqClsDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        input_mask,
        label_ids,
        label_size=2,
        sample_weight=None,
        scope="cls/sequence",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if kwargs.get("is_logits"):
            logits = input_tensor
        else:
            shape = input_tensor.shape.as_list()
            seq_length = shape[-2]
            hidden_size = shape[-1]
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
                output_layer = tf.reshape(output_layer, [-1, hidden_size])
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                logits = tf.reshape(logits, [-1, seq_length, label_size])

        self.tensors["preds"] = tf.argmax(logits, axis=-1, name="preds")
        self.tensors["probs"] = tf.nn.softmax(logits, axis=-1, name="probs")

        per_token_loss = util.cross_entropy(logits, label_ids, label_size, **kwargs)
        input_mask = tf.cast(input_mask, tf.float32)
        per_token_loss *= input_mask / tf.reduce_sum(input_mask, keepdims=True, axis=-1)
        per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
        if sample_weight is not None:
            per_example_loss *= sample_weight

        self.tensors["losses"] = per_example_loss
        self.train_loss = tf.reduce_mean(per_example_loss)


class SeqClsCrossDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        input_mask,
        seq_cls_label_ids,
        cls_label_ids,
        seq_cls_label_size=2,
        cls_label_size=2,
        sample_weight=None,
        seq_cls_scope="cls/tokens",
        cls_scope="cls/sequence",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert not kwargs.get("is_logits"), "%s does not support logits convertion right now." % self.__class__.__name__

        shape = input_tensor.shape.as_list()
        seq_length = shape[-2]
        hidden_size = shape[-1]

        # seq cls
        with tf.variable_scope(seq_cls_scope):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[seq_cls_label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable,
            )
            output_bias = tf.get_variable(
                "output_bias",
                shape=[seq_cls_label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable,
            )
            output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, seq_length, seq_cls_label_size])
            self.tensors["seq_cls_preds"] = tf.argmax(logits, axis=-1, name="seq_cls_preds")
            self.tensors["seq_cls_probs"] = tf.nn.softmax(logits, axis=-1, name="seq_cls_probs")
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(seq_cls_label_ids, depth=seq_cls_label_size, dtype=tf.float32)
            per_token_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            input_mask = tf.cast(input_mask, tf.float32)
            per_token_loss *= input_mask / tf.reduce_sum(input_mask, keepdims=True, axis=-1)
            per_example_loss = tf.reduce_sum(per_token_loss, axis=-1)
            if sample_weight is not None:
                per_example_loss *= sample_weight
            self.tensors["seq_cls_losses"] = per_example_loss

        # cls
        with tf.variable_scope(cls_scope):
            output_weights = tf.get_variable(
                "output_weights",
                shape=[cls_label_size, hidden_size],
                initializer=util.create_initializer(initializer_range),
                trainable=trainable,
            )
            output_bias = tf.get_variable(
                "output_bias",
                shape=[cls_label_size],
                initializer=tf.zeros_initializer(),
                trainable=trainable,
            )
            output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
            logits = tf.matmul(output_layer[:,0,:], output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            self.tensors["cls_preds"] = tf.argmax(logits, axis=-1, name="cls_preds")
            self.tensors["cls_probs"] = tf.nn.softmax(logits, axis=-1, name="cls_probs")
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(cls_label_ids, depth=cls_label_size, dtype=tf.float32)
            per_example_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            if sample_weight is not None:
                per_example_loss *= sample_weight
            self.tensors["cls_losses"] = per_example_loss

        self.train_loss = tf.reduce_mean(self.tensors["seq_cls_losses"]) + tf.reduce_mean(self.tensors["cls_losses"])


class SeqClassifierModule(BaseModule):
    """ Application class of classification. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def _convert_x(self, x, tokenized):
        if not tokenized:
            raise ValueError("Inputs of sequence classifier must be already tokenized and fed into `X_tokenized`.")

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return x

        # deal with tokenized and multiple inputs
        raise ValueError("Sequence classifier does not support multi-segment inputs.")

    def _convert_y(self, y):
        try:
            label_set = set()
            for sample in y:
                for _y in sample:
                    label_set.add(_y)
        except Exception:
            raise ValueError("The element of `y` should be a list of labels.")

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
        for sample in y:
            sample = [label for label in sample]

            num_labels = len(sample)
            if num_labels < self.max_seq_length:
                sample.extend([0] * (self.max_seq_length - num_labels))
            elif num_labels > self.max_seq_length:
                sample = sample[:self.max_seq_length]

                com.truncate_segments([sample], self.max_seq_length, truncate_method=self.truncate_method)

            _label_ids = []
            for label in sample:
                if label not in self._label_to_id:
                    assert len(self._label_to_id) < self.label_size, "Number of unique labels exceeds `label_size`."
                    self._label_to_id[label] = len(self._label_to_id)
                    self._id_to_label.append(label)
                _label_ids.append(self._label_to_id[label])
            label_ids.append(_label_ids)
        return label_ids

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # accuracy
        batch_preds = output_arrays[0]
        accuracy = (np.sum((batch_preds == batch_labels) * batch_mask) / batch_mask.sum())

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
        all_preds = np.argmax(probs, axis=-1)
        mask = self.data["input_mask"]
        preds = []
        for _preds, _mask in zip(all_preds, mask):
            input_length = np.sum(_mask)
            _preds = _preds[:input_length].tolist()
            if self._id_to_label:
                _preds = [self._id_to_label[idx] if idx < len(self._id_to_label) else None for idx in _preds]
            preds.append(_preds)

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
        mask = self.data["input_mask"]
        accuracy = (np.sum((preds == labels) * mask) / mask.sum())

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["accuracy"] = accuracy
        outputs["loss"] = loss

        return outputs
