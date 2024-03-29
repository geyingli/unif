import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .._base_._base_seq_classifier import SeqClsCrossDecoder, SeqClassifierModule
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTSeqCrossClassifier(SeqClassifierModule):
    """ Sequence labeling and single-label (multi-task) classifier on BERT. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "seq_cls_label_size": "An integer that defines number of possible labels of sequence labeling",
        "cls_label_size": "An integer that defines number of possible labels of sequence classification",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }
    _LOCALIZE_ATTRIBUTES = [    # attributes to store in localization
        "_seq_cls_id_to_label",
        "_seq_cls_label_to_id",
        "_cls_id_to_label",
        "_cls_label_to_id",
    ]

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        seq_cls_label_size=None,
        cls_label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(SeqClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.seq_cls_label_size = seq_cls_label_size
        self.cls_label_size = cls_label_size
        self.truncate_method = truncate_method
        self._seq_cls_id_to_label = None
        self._seq_cls_label_to_id = None
        self._cls_id_to_label = None
        self._cls_label_to_id = None

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
            assert self.seq_cls_label_size, "Can't parse data on multi-processing when `seq_cls_label_size` is None."
            assert self.cls_label_size, "Can't parse data on multi-processing when `cls_label_size` is None."

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
            seq_cls_label_ids, cls_label_ids = self._convert_y(y)
            data["seq_cls_label_ids"] = np.array(seq_cls_label_ids, dtype=np.int32)
            data["cls_label_ids"] = np.array(cls_label_ids, dtype=np.int32)

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
            _input_tokens = ["CLS"] + self._convert_x(sample, tokenized)

            com.truncate_segments([_input_tokens], self.max_seq_length, truncate_method=self.truncate_method)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _input_mask = [0] + [1 for _ in range(len(_input_tokens) - 1)]
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

    def _convert_y(self, y):
        try:
            seq_cls_label_set = set()
            cls_label_set = set()
            for sample in y:
                for _y in sample["seq_cls"]:
                    seq_cls_label_set.add(_y)
                cls_label_set.add(sample["cls"])
        except Exception:
            raise ValueError("The element of `y` should be a list of seq_cls labels and cls labels. An example: X = [{\"seq_cls\": [0, 1, 0, 0, 2, 4, 0], \"cls\": 1}, ...]")

        # automatically set `label_size`
        if self.seq_cls_label_size:
            assert len(seq_cls_label_set) <= self.seq_cls_label_size, ("Number of unique `y`s exceeds `seq_cls_label_size`.")
        else:
            self.seq_cls_label_size = len(seq_cls_label_set)
        if self.cls_label_size:
            assert len(cls_label_set) <= self.cls_label_size, ("Number of unique `y`s exceeds `cls_label_size`.")
        else:
            self.cls_label_size = len(cls_label_set)

        # automatically set `id_to_label`
        if not self._seq_cls_id_to_label:
            self._seq_cls_id_to_label = list(seq_cls_label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._seq_cls_id_to_label = list(sorted(self._seq_cls_id_to_label))
            except Exception:
                pass
        if not self._cls_id_to_label:
            self._cls_id_to_label = list(cls_label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._cls_id_to_label = list(sorted(self._cls_id_to_label))
            except Exception:
                pass

        # automatically set `label_to_id` for prediction
        if not self._seq_cls_label_to_id:
            self._seq_cls_label_to_id = {label: index for index, label in enumerate(self._seq_cls_id_to_label)}
        if not self._cls_label_to_id:
            self._cls_label_to_id = {label: index for index, label in enumerate(self._cls_id_to_label)}

        seq_cls_label_ids = []
        cls_label_ids = []
        for sample in y:
            _seq_cls_label_ids = []

            for label in sample["seq_cls"]:
                if label not in self._seq_cls_label_to_id:
                    assert len(self._seq_cls_label_to_id) < self.seq_cls_label_size, "Number of unique labels exceeds `label_size`."
                    self._seq_cls_label_to_id[label] = len(self._seq_cls_label_to_id)
                    self._seq_cls_id_to_label.append(label)
                _seq_cls_label_ids.append(self._seq_cls_label_to_id[label])

            com.truncate_segments([_seq_cls_label_ids], self.max_seq_length - 1, truncate_method=self.truncate_method)
            _seq_cls_label_ids.insert(0, 0)
            for _ in range(self.max_seq_length - len(_seq_cls_label_ids)):
                _seq_cls_label_ids.append(0)
            seq_cls_label_ids.append(_seq_cls_label_ids)

            label = sample["cls"]
            if label not in self._cls_label_to_id:
                assert len(self._cls_label_to_id) < self.cls_label_size, "Number of unique labels exceeds `label_size`."
                self._cls_label_to_id[label] = len(self._cls_label_to_id)
                self._cls_id_to_label.append(label)
            cls_label_ids.append(self._cls_label_to_id[label])
        return seq_cls_label_ids, cls_label_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "seq_cls_label_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "seq_cls_label_ids"),
            "cls_label_ids": tf.placeholder(tf.int32, [None], "cls_label_ids"),
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
        decoder = SeqClsCrossDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=placeholders["input_mask"],
            seq_cls_label_ids=placeholders["seq_cls_label_ids"],
            cls_label_ids=placeholders["cls_label_ids"],
            seq_cls_label_size=self.seq_cls_label_size,
            cls_label_size=self.cls_label_size,
            sample_weight=placeholders.get("sample_weight"),
            seq_cls_scope="cls/tokens",
            cls_scope="cls/sequence",
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [
            self.tensors["seq_cls_preds"],
            self.tensors["cls_preds"],
            self.tensors["seq_cls_losses"],
            self.tensors["cls_losses"],
        ]
        if from_tfrecords:
            ops.extend([
                self.placeholders["input_mask"],
                self.placeholders["seq_cls_label_ids"],
                self.placeholders["cls_label_ids"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mask = output_arrays[-3]
            batch_seq_cls_labels = output_arrays[-2]
            batch_cls_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_seq_cls_labels = feed_dict[self.placeholders["seq_cls_label_ids"]]
            batch_cls_labels = feed_dict[self.placeholders["cls_label_ids"]]

        # accuracy
        batch_seq_cls_preds = output_arrays[0]
        batch_cls_preds = output_arrays[1]
        seq_cls_accuracy = (np.sum((batch_seq_cls_preds == batch_seq_cls_labels) * batch_mask) / batch_mask.sum())
        cls_accuracy = np.mean(batch_cls_preds == batch_cls_labels)

        # loss
        batch_seq_cls_losses = output_arrays[2]
        batch_cls_losses = output_arrays[3]
        seq_cls_loss = np.mean(batch_seq_cls_losses)
        cls_loss = np.mean(batch_cls_losses)

        info = ""
        info += ", seq_cls_accuracy %.4f" % seq_cls_accuracy
        info += ", cls_accuracy %.4f" % cls_accuracy
        info += ", seq_cls_loss %.6f" % seq_cls_loss
        info += ", cls_loss %.6f" % cls_loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["seq_cls_probs"], self.tensors["cls_probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        seq_cls_probs = com.transform(output_arrays[0], n_inputs)
        cls_probs = com.transform(output_arrays[1], n_inputs)

        # preds
        seq_cls_preds = []
        seq_cls_all_preds = np.argmax(seq_cls_probs, axis=-1)
        for _preds, _mask in zip(seq_cls_all_preds, self.data["input_mask"]):
            _preds = [idx for i, idx in enumerate(_preds) if _mask[i] > 0]
            if self._seq_cls_id_to_label:
                _preds = [self._seq_cls_id_to_label[idx] if idx < len(self._seq_cls_id_to_label) else None for idx in _preds]
            seq_cls_preds.append(_preds)
        cls_preds = np.argmax(cls_probs, axis=-1).tolist()
        if self._cls_id_to_label:
            cls_preds = [self._cls_id_to_label[idx] if idx < len(self._cls_id_to_label) else None for idx in cls_preds]

        outputs = {}
        outputs["seq_cls_preds"] = seq_cls_preds
        outputs["seq_cls_probs"] = seq_cls_probs
        outputs["cls_preds"] = cls_preds
        outputs["cls_probs"] = cls_probs

        return outputs

    def _get_score_ops(self):
        return [
            self.tensors["seq_cls_preds"],
            self.tensors["cls_preds"],
            self.tensors["seq_cls_losses"],
            self.tensors["cls_losses"],
        ]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # accuracy
        seq_cls_preds = com.transform(output_arrays[0], n_inputs)
        cls_preds = com.transform(output_arrays[1], n_inputs)
        seq_cls_labels = self.data["seq_cls_label_ids"]
        cls_labels = self.data["cls_label_ids"]
        mask = self.data["input_mask"]
        seq_cls_accuracy = (np.sum((seq_cls_preds == seq_cls_labels) * mask) / mask.sum())
        cls_accuracy = np.mean(cls_preds == cls_labels)

        # loss
        seq_cls_losses = com.transform(output_arrays[2], n_inputs)
        cls_losses = com.transform(output_arrays[3], n_inputs)
        seq_cls_loss = np.mean(seq_cls_losses)
        cls_loss = np.mean(cls_losses)

        outputs = {}
        outputs["seq_cls_accuracy"] = seq_cls_accuracy
        outputs["cls_accuracy"] = cls_accuracy
        outputs["seq_cls_loss"] = seq_cls_loss
        outputs["cls_loss"] = cls_loss

        return outputs
