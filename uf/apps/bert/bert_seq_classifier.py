import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_classifier import BERTClassifier
from .._base_._base_classifier import ClassifierModule
from .._base_._base_ import SeqClsDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTSeqClassifier(BERTClassifier, ClassifierModule):
    """ Sequence labeling classifier on BERT. """

    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._id_to_label = None

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(self.bert_config.num_hidden_layers)

        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            self.bert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            self.bert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."
        if is_parallel:
            assert self.label_size, "Can't parse data on multi-processing when `label_size` is None."

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y is not None:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        input_ids = []
        input_mask = []
        segment_ids = []

        # tokenize input texts
        for idx, sample in enumerate(X_target):
            _input_tokens = self._convert_x(sample, tokenized)

            com.truncate_segments([_input_tokens], self.max_seq_length, truncate_method=self.truncate_method)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _input_mask = [1 for _ in range(len(_input_tokens))]
            _segment_ids = [0 for _ in range(len(_input_tokens))]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

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
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
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

            _label_ids = [self._label_to_id[label] for label in sample]
            label_ids.append(_label_ids)
        return label_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = SeqClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=placeholders["input_mask"],
            label_ids=placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
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
        return [self._tensors["probs"]]

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
                _preds = [self._id_to_label[idx] for idx in _preds]
            preds.append(_preds)

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
        mask = self.data["input_mask"]
        accuracy = (np.sum((preds == labels) * mask) / mask.sum())

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["accuracy"] = accuracy
        outputs["loss"] = loss

        return outputs
