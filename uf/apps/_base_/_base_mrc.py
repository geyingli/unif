import numpy as np

from ._base_ import BaseDecoder
from ...core import BaseModule
from ...third import tf
from ... import com
from .. import util


class MRCDecoder(BaseDecoder):
    def __init__(
        self,
        is_training,
        input_tensor,
        label_ids,
        is_logits=False,
        sample_weight=None,
        scope="mrc",
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        trainable=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if is_logits:
            logits = input_tensor
        else:
            seq_length = input_tensor.shape.as_list()[-2]
            hidden_size = input_tensor.shape.as_list()[-1]
            with tf.variable_scope(scope):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[2, hidden_size],
                    initializer=util.create_initializer(initializer_range),
                    trainable=trainable,
                )
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[2],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable,
                )
                output_layer = util.dropout(input_tensor, hidden_dropout_prob if is_training else 0.0)
                output_layer = tf.reshape(output_layer, [-1, hidden_size])
                logits = tf.matmul(output_layer, output_weights, transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                logits = tf.reshape(logits, [-1, seq_length, 2])
                logits = tf.transpose(logits, [0, 2, 1])

        probs = tf.nn.softmax(logits, axis=-1, name="probs")
        self._tensors["probs"] = probs
        self._tensors["preds"] = tf.argmax(logits, axis=-1, name="preds")

        start_one_hot_labels = tf.one_hot(label_ids[:, 0], depth=seq_length, dtype=tf.float32)
        end_one_hot_labels = tf.one_hot(label_ids[:, 1], depth=seq_length, dtype=tf.float32)
        start_log_probs = tf.nn.log_softmax(logits[:, 0, :], axis=-1)
        end_log_probs = tf.nn.log_softmax(logits[:, 1, :], axis=-1)
        per_example_loss = (
            - 0.5 * tf.reduce_sum(start_one_hot_labels * start_log_probs, axis=-1)
            - 0.5 * tf.reduce_sum(end_one_hot_labels * end_log_probs, axis=-1)
        )
        if sample_weight is not None:
            per_example_loss *= sample_weight

        self._tensors["losses"] = per_example_loss
        self.train_loss = tf.reduce_mean(per_example_loss)


class MRCModule(BaseModule):
    """ Application class of machine reading comprehension (MRC). """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def _get_em_and_f1(self, preds, labels):
        """ Exact-match and F1. """
        em, f1 = 0, 0
        for _preds, _labels in zip(preds, labels):
            start_pred, end_pred = int(_preds[0]), int(_preds[1])
            start_label, end_label = int(_labels[0]), int(_labels[1])

            # no answer prediction
            if start_pred == 0 or end_pred == 0 or start_pred > end_pred:
                if start_label == 0:
                    em += 1
                    f1 += 1

            # answer prediction (no intersection)
            elif start_pred > end_label or end_pred < start_label:
                pass

            # answer prediction (has intersection)
            else:
                tp = (min(end_pred, end_label) + 1 - max(start_pred, start_label))
                fp = (max(0, end_pred - end_label) + max(0, start_label - start_pred))
                fn = (max(0, start_pred - start_label) + max(0, end_label - end_pred))
                if fp + fn == 0:
                    em += 1
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 += 2 * precision * recall / max(precision + recall, 1)

        em /= len(labels)
        f1 /= len(labels)
        return em, f1

    def _convert_x(self, x, tokenized):
        output = {}

        assert isinstance(x, dict) and "doc" in x

        for key in x:
            if not tokenized:
                # deal with general inputs
                output[key] = self.tokenizer.tokenize(x[key])
                continue

            # deal with tokenized inputs
            output[key] = x[key]

        return output

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []

        invalid_ids = []
        for idx, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                continue

            if not isinstance(_y, dict) or "text" not in _y or "answer_start" not in _y:
                raise ValueError(
                    "Wrong label format (%s). An untokenized example: "
                    "`y = [{\"text\": \"Obama\", \"answer_start\": 12}, "
                    "None, ...]`"
                    % (_y)
                )

            _answer_text = _y["text"]
            _answer_start = _y["answer_start"]
            _doc_ids = doc_ids[idx]

            # tokenization
            if isinstance(_answer_text, str):
                _answer_tokens = self.tokenizer.tokenize(_answer_text)
                _answer_ids = self.tokenizer.convert_tokens_to_ids(_answer_tokens)
            elif isinstance(_answer_text, list):
                assert tokenized, "%s does not support multiple answer spans." % self.__class__.__name__
                _answer_tokens = _answer_text
                _answer_ids = self.tokenizer.convert_tokens_to_ids(_answer_text)
            else:
                raise ValueError("Invalid answer text at line %d: `%s`." % (idx, _answer_text))

            if isinstance(_answer_text, str):
                _overlap_time = len(com.find_all_boyer_moore(doc_text[idx][:_answer_start], _answer_text))
                try:
                    start_position = com.find_all_boyer_moore(_doc_ids, _answer_ids)[_overlap_time]
                    end_position = start_position + len(_answer_ids) - 1
                except IndexError:
                    label_ids.append([0, 0])
                    invalid_ids.append(idx)
                    continue
            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning("Wrong `answer_start` at line %d. Ignored and set label to null text.")
                    label_ids.append([0, 0])
                    invalid_ids.append(idx)
                    continue

            label_ids.append([start_position + doc_start[idx], end_position + doc_start[idx]])

        if invalid_ids:
            tf.logging.warning(
                "Failed to find the mapping of answer to inputs at "
                "line %s. A possible reason is that the answer spans "
                "are truncated due to the `max_seq_length` setting."
                % invalid_ids
            )

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

        # exact match & f1
        batch_preds = output_arrays[0]
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", exact_match %.4f" % exact_match
        info += ", f1 %.4f" % f1
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["probs"], self._tensors["preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        batch_preds = com.transform(output_arrays[1], n_inputs)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for idx, _preds in enumerate(batch_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if _start == 0 or _end == 0 or _start > _end:
                preds.append(None)
                continue
            _tokens = tokens[idx]

            if tokenized:
                _span_tokens = _tokens[_start: _end + 1]
                preds.append(_span_tokens)
            else:
                _sample = text[idx]
                _text = [_sample[key] for key in _sample if key != "doc"]
                _text.append(_sample["doc"])
                _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

                try:
                    _text_start = _mapping_start[_start]
                    _text_end = _mapping_end[_end]
                except Exception:
                    preds.append(None)
                    continue
                _span_text = _text[_text_start: _text_end]
                preds.append(_span_text)

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["preds"], self._tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # exact match & f1
        batch_preds = com.transform(output_arrays[0], n_inputs)
        batch_labels = self.data["label_ids"]
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["exact_match"] = exact_match
        outputs["f1"] = f1
        outputs["loss"] = loss

        return outputs
