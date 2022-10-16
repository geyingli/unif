
import collections
import numpy as np

from .base import ClassifierModule, LMModule, NERModule, MRCModule
from ..model.base import ClsDecoder, BinaryClsDecoder, SeqClsDecoder, SeqClsCrossDecoder, MRCDecoder
from ..model.bert import BERTEncoder, BERTDecoder, BERTConfig, create_instances_from_document, create_masked_lm_predictions, get_decay_power
from ..model.crf import CRFDecoder, viterbi_decode
from ..token import WordPieceTokenizer
from ..third import tf
from .. import com


class BERTClassifier(ClassifierModule):
    """ Single-label classifier on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
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
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception:
                raise ValueError("Wrong input format (line %d): \"%s\". " % (idx, sample))

        input_ids = []
        input_mask = []
        segment_ids = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

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
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {label: index for index, label in enumerate(self._id_to_label)}

        label_ids = [self._label_to_id[label] for label in y]
        return label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()
        decoder = ClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
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

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        preds = np.argmax(probs, axis=-1).tolist()
        if self._id_to_label:
            preds = [self._id_to_label[idx] for idx in preds]

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["preds"], self._tensors["losses"]]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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


class BERTBinaryClassifier(BERTClassifier, ClassifierModule):
    """ Multi-label classifier on BERT. """
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        label_weight=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.label_weight = label_weight
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
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
            if len(self._id_to_label) < self.label_size:
                self._id_to_label = list(range(self.label_size))

        # automatically set `label_to_id` for prediction
        self._label_to_id = {label: index for index, label in enumerate(self._id_to_label)}

        label_ids = [[1 if self._id_to_label[i] in sample else 0 for i in range(self.label_size)] for sample in y]
        return label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, self.label_size], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()
        decoder = BinaryClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            label_weight=self.label_weight,
            scope="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_predict_ops(self):
        return [self._tensors["probs"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))
        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        preds = (probs >= 0.5)
        if self._id_to_label:
            preds = [[self._id_to_label[i] for i in range(self.label_size) if _preds[i]] for _preds in preds]
        else:
            preds = [[i for i in range(self.label_size) if _preds[i]] for _preds in preds]

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs


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

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."
        if is_parallel:
            assert self.label_size, "Can't parse data on multi-processing when `label_size` is None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
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

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, self.max_seq_length], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = SeqClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            label_ids=split_placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
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

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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


class BERTSeqCrossClassifier(BERTClassifier, ClassifierModule):
    """ Sequence labeling and single-label (multi-task) classifier on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "seq_cls_label_size": "An integer that defines number of possible labels of sequence labeling",
        "cls_label_size": "An integer that defines number of possible labels of sequence classification",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

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
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.seq_cls_label_size = seq_cls_label_size
        self.cls_label_size = cls_label_size
        self.truncate_method = truncate_method
        self._seq_cls_id_to_label = None
        self._cls_id_to_label = None

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(self.bert_config.num_hidden_layers)

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
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            seq_cls_label_ids, cls_label_ids = self._convert_y(y)
            data["seq_cls_label_ids"] = np.array(seq_cls_label_ids, dtype=np.int32)
            data["cls_label_ids"] = np.array(cls_label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
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
            if len(self._seq_cls_id_to_label) < self.seq_cls_label_size:
                self._seq_cls_id_to_label = list(range(self.seq_cls_label_size))
        if not self._cls_id_to_label:
            self._cls_id_to_label = list(cls_label_set)
            try:
                # Allign if user inputs continual integers.
                # e.g. [2, 0, 1]
                self._cls_id_to_label = list(sorted(self._cls_id_to_label))
            except Exception:
                pass
            if len(self._cls_id_to_label) < self.cls_label_size:
                self._cls_id_to_label = list(range(self.cls_label_size))

        # automatically set `label_to_id` for prediction
        self._seq_cls_label_to_id = {label: index for index, label in enumerate(self._seq_cls_id_to_label)}
        self._cls_label_to_id = {label: index for index, label in enumerate(self._cls_id_to_label)}

        seq_cls_label_ids = []
        cls_label_ids = []
        for sample in y:
            _labels = [label for label in sample["seq_cls"]]
            _label_ids = [self._seq_cls_label_to_id[label] for label in _labels]
            com.truncate_segments([_label_ids], self.max_seq_length - 1, truncate_method=self.truncate_method)
            _label_ids.insert(0, 0)
            for _ in range(self.max_seq_length - len(_label_ids)):
                _label_ids.append(0)
            seq_cls_label_ids.append(_label_ids)
        for sample in y:
            _label = sample["cls"]
            cls_label_ids.append(self._cls_label_to_id[_label])

        return seq_cls_label_ids, cls_label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "seq_cls_label_ids": com.get_placeholder(target, "seq_cls_label_ids", [None, self.max_seq_length], tf.int32),
            "cls_label_ids": com.get_placeholder(target, "cls_label_ids", [None], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = SeqClsCrossDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            seq_cls_label_ids=split_placeholders["seq_cls_label_ids"],
            cls_label_ids=split_placeholders["cls_label_ids"],
            seq_cls_label_size=self.seq_cls_label_size,
            cls_label_size=self.cls_label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            seq_cls_scope="cls/tokens",
            cls_scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [
            self._tensors["seq_cls_preds"],
            self._tensors["cls_preds"],
            self._tensors["seq_cls_losses"],
            self._tensors["cls_losses"],
        ]
        if as_feature:
            ops.extend([
                self.placeholders["input_mask"],
                self.placeholders["seq_cls_label_ids"],
                self.placeholders["cls_label_ids"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
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
        return [self._tensors["seq_cls_probs"], self._tensors["cls_probs"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        seq_cls_probs = com.transform(output_arrays[0], n_inputs)
        cls_probs = com.transform(output_arrays[1], n_inputs)

        # preds
        seq_cls_preds = []
        seq_cls_all_preds = np.argmax(seq_cls_probs, axis=-1)
        for _preds, _mask in zip(seq_cls_all_preds, self.data["input_mask"]):
            _preds = [idx for i, idx in enumerate(_preds) if _mask[i] > 0]
            if self._seq_cls_id_to_label:
                _preds = [self._seq_cls_id_to_label[idx] for idx in _preds]
            seq_cls_preds.append(_preds)
        cls_preds = np.argmax(cls_probs, axis=-1).tolist()
        if self._cls_id_to_label:
            cls_preds = [self._cls_id_to_label[idx] for idx in cls_preds]

        outputs = {}
        outputs["seq_cls_preds"] = seq_cls_preds
        outputs["seq_cls_probs"] = seq_cls_probs
        outputs["cls_preds"] = cls_preds
        outputs["cls_probs"] = cls_probs

        return outputs

    def _get_score_ops(self):
        return [
            self._tensors["seq_cls_preds"],
            self._tensors["cls_preds"],
            self._tensors["seq_cls_losses"],
            self._tensors["cls_losses"],
        ]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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


class BERTSeqCrossTmpClassifier(BERTClassifier, ClassifierModule):
    """ Sequence labeling and single-label (multi-task) classifier on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "seq_cls_labels": "An integer that defines number of possible labels of sequence labeling",
        "cls_labels": "An integer that defines number of possible labels of sequence classification",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        seq_cls_labels=None,
        cls_labels=None,
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
        self.seq_cls_labels = seq_cls_labels
        self.cls_labels = cls_labels
        self.truncate_method = truncate_method

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(self.bert_config.num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."
        if is_parallel:
            assert self.seq_cls_labels, "Can't parse data on multi-processing when `seq_cls_labels` is None."
            assert self.cls_labels, "Can't parse data on multi-processing when `cls_labels` is None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            seq_cls_label_ids, cls_label_ids = self._convert_y(y)
            data["seq_cls_label_ids"] = np.array(seq_cls_label_ids, dtype=np.int32)
            data["cls_label_ids"] = np.array(cls_label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
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

            com.truncate_segments([_input_tokens], self.max_seq_length - 2, truncate_method=self.truncate_method)

            _input_tokens = ["[CLS]"] + _input_tokens + ["[SEP]"]
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
            seq_cls_label_set = {"[CLS]", "[SEP]"}
            cls_label_set = set()
            for sample in y:
                for _y in sample["seq_cls"]:
                    seq_cls_label_set.add(_y)
                cls_label_set.add(sample["cls"])
        except Exception:
            raise ValueError("The element of `y` should be a list of seq_cls labels and cls labels. An example: X = [{\"seq_cls\": [0, 1, 0, 0, 2, 4, 0], \"cls\": 1}, ...]")

        # automatically set `label_size`
        if self.seq_cls_labels:
            if "[CLS]" not in self.seq_cls_labels:
                self.seq_cls_labels.append("[CLS]")
            if "[SEP]" not in self.seq_cls_labels:
                self.seq_cls_labels.append("[SEP]")
            assert len(seq_cls_label_set) <= len(self.seq_cls_labels), ("Number of unique `y`s exceeds `seq_cls_labels`s.")
        else:
            self.seq_cls_labels = list(seq_cls_label_set)
        try:
            self.seq_cls_labels.sort()
        except:
            pass
        if self.cls_labels:
            assert len(cls_label_set) <= len(self.cls_labels), ("Number of unique `y`s exceeds `cls_labels`s.")
        else:
            self.cls_labels = list(cls_label_set)
        try:
            self.cls_labels.sort()
        except:
            pass

        # automatically set `label_to_id` for prediction
        self._seq_cls_label_to_id = {label: index for index, label in enumerate(self.seq_cls_labels)}
        self._cls_label_to_id = {label: index for index, label in enumerate(self.cls_labels)}

        seq_cls_label_ids = []
        cls_label_ids = []
        for sample in y:
            _labels = [label for label in sample["seq_cls"]]
            com.truncate_segments([_labels], self.max_seq_length - 2, truncate_method=self.truncate_method)
            _labels = ["[CLS]"] + _labels + ["[SEP]"]
            _label_ids = [self._seq_cls_label_to_id[label] for label in _labels]
            for _ in range(self.max_seq_length - len(_label_ids)):
                _label_ids.append(0)
            seq_cls_label_ids.append(_label_ids)
        for sample in y:
            _label = sample["cls"]
            cls_label_ids.append(self._cls_label_to_id[_label])

        return seq_cls_label_ids, cls_label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "seq_cls_label_ids": com.get_placeholder(target, "seq_cls_label_ids", [None, self.max_seq_length], tf.int32),
            "cls_label_ids": com.get_placeholder(target, "cls_label_ids", [None], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = SeqClsCrossDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            seq_cls_label_ids=split_placeholders["seq_cls_label_ids"],
            cls_label_ids=split_placeholders["cls_label_ids"],
            seq_cls_label_size=len(self.seq_cls_labels),
            cls_label_size=len(self.cls_labels),
            sample_weight=split_placeholders.get("sample_weight"),
            seq_cls_scope="cls/tokens",
            cls_scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [
            self._tensors["seq_cls_preds"],
            self._tensors["cls_preds"],
            self._tensors["seq_cls_losses"],
            self._tensors["cls_losses"],
        ]
        if as_feature:
            ops.extend([
                self.placeholders["input_mask"],
                self.placeholders["seq_cls_label_ids"],
                self.placeholders["cls_label_ids"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
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
        return [self._tensors["seq_cls_probs"], self._tensors["cls_probs"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        seq_cls_probs = com.transform(output_arrays[0], n_inputs)
        cls_probs = com.transform(output_arrays[1], n_inputs)

        # preds
        seq_cls_preds = []
        seq_cls_all_preds = np.argmax(seq_cls_probs, axis=-1)
        for _preds, _mask in zip(seq_cls_all_preds, self.data["input_mask"]):
            _preds = [idx for i, idx in enumerate(_preds) if _mask[i] > 0]
            if self.seq_cls_labels:
                _preds = [self.seq_cls_labels[idx] for idx in _preds]
            seq_cls_preds.append(_preds)
        cls_preds = np.argmax(cls_probs, axis=-1).tolist()
        if self.cls_labels:
            cls_preds = [self.cls_labels[idx] for idx in cls_preds]

        outputs = {}
        outputs["seq_cls_preds"] = seq_cls_preds
        outputs["seq_cls_probs"] = seq_cls_probs
        outputs["cls_preds"] = cls_preds
        outputs["cls_probs"] = cls_probs

        return outputs

    def _get_score_ops(self):
        return [
            self._tensors["seq_cls_preds"],
            self._tensors["cls_preds"],
            self._tensors["seq_cls_losses"],
            self._tensors["cls_losses"],
        ]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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


class BERTNER(BERTClassifier, NERModule):
    """ Named entity recognition on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(NERModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._on_predict = False

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

    def predict(self, X=None, X_tokenized=None, batch_size=8):

        self._on_predict = True
        ret = super(NERModule, self).predict(X, X_tokenized, batch_size)
        self._on_predict = False

        return ret

    predict.__doc__ = NERModule.predict.__doc__

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            input_tokens, input_ids, input_mask, segment_ids = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception:
                raise ValueError("Wrong input format (line %d): \"%s\". " % (idx, sample))

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_tokens, input_ids, input_mask, segment_ids

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        for idx, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if isinstance(_y, str):
                _entity_tokens = self.tokenizer.tokenize(_y)
                _entity_ids = [self.tokenizer.convert_tokens_to_ids(_entity_tokens)]
            elif isinstance(_y, list):
                if isinstance(_y[0], str):
                    if tokenized:
                        _entity_ids = [self.tokenizer.convert_tokens_to_ids(_y)]
                    else:
                        _entity_ids = []
                        for _entity in _y:
                            _entity_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_entity)))
                elif isinstance(_y[0], list):
                    _entity_ids = []
                    for _entity in _y:
                        _entity_ids.append(self.tokenizer.convert_tokens_to_ids(_entity))
            else:
                raise ValueError("`y` should be a list of entity strings.")

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]
            for _entity in _entity_ids:
                start_positions = com.find_all_boyer_moore(_input_ids, _entity)
                if not start_positions:
                    tf.logging.warning(
                        "Failed to find the mapping of entity to "
                        "inputs at line %d. A possible reason is "
                        "that the entity span is truncated due "
                        "to the `max_seq_length` setting." % (idx)
                    )
                    continue

                for start_position in start_positions:
                    end_position = start_position + len(_entity) - 1
                    if start_position == end_position:
                        _label_ids[start_position] = self.S_ID
                    else:
                        for i in range(start_position, end_position + 1):
                            _label_ids[i] = self.I_ID
                        _label_ids[start_position] = self.B_ID
                        _label_ids[end_position] = self.E_ID

            label_ids.append(_label_ids)

        return label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, self.max_seq_length], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = SeqClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            label_ids=split_placeholders["label_ids"],
            label_size=5,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_preds = output_arrays[0]
        f1_token, f1_entity = self._get_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", f1/token %.4f" % f1_token
        info += ", f1/entity %.4f" % f1_entity
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["probs"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        all_preds = np.argmax(probs, axis=-1)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(all_preds)):
            _preds = all_preds[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _entities = self._get_entities(_preds[:_input_length])
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except Exception:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["preds"], self._tensors["losses"]]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        preds = com.transform(output_arrays[0], n_inputs)
        labels = self.data["label_ids"]
        mask = self.data["input_mask"]
        f1_token, f1_entity = self._get_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["f1/token"] = f1_token
        outputs["f1/entity"] = f1_entity
        outputs["loss"] = loss

        return outputs


class BERTCRFNER(BERTNER, NERModule):
    """ Named entity recognization on BERT with CRF. """
    _INFER_ATTRIBUTES = BERTNER._INFER_ATTRIBUTES

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            label_ids=split_placeholders["label_ids"],
            label_size=5,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_logits = output_arrays[0]
        batch_transition_matrix = output_arrays[1]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ""
        info += ", f1/token %.4f" % f1_token
        info += ", f1/entity %.4f" % f1_entity
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(_logits[:_input_length], transition_matrix)
            _entities = self._get_entities(_viterbi_seq)
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except Exception:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs["preds"] = preds
        outputs["logits"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data["input_mask"]
        labels = self.data["label_ids"]
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["f1/token"] = f1_token
        outputs["f1/entity"] = f1_entity
        outputs["loss"] = loss

        return outputs


class BERTCRFCascadeNER(BERTCRFNER, NERModule):
    """ Named entity recognization and classification on BERT with CRF. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "entity_types": "A list of strings that defines possible types of entities",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        entity_types=None,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(NERModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.entity_types = entity_types
        self._do_lower_case = do_lower_case
        self._on_predict = False

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
            assert self.entity_types, "Can't parse data on multi-processing when `entity_types` is None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            input_tokens, input_ids, input_mask, segment_ids = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        if not self.entity_types:
            type_B_id = {}
        else:
            type_B_id = {
                entity_type: 1 + 4 * i
                for i, entity_type in enumerate(self.entity_types)
            }
        for idx, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if not isinstance(_y, dict):
                raise ValueError(
                    "Wrong input format of `y`. An untokenized example: "
                    "`y = [{\"Person\": [\"Trump\", \"Obama\"], "
                    "\"City\": [\"Washington D.C.\"], ...}, ...]`"
                )

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]

            # each type
            for _key in _y:

                # new type
                if _key not in type_B_id:
                    assert not self.entity_types, (
                        "Entity type `%s` not found in entity_types: %s."
                        % (_key, self.entity_types)
                    )
                    type_B_id[_key] = 1 + 4 * len(list(type_B_id.keys()))
                _entities = _y[_key]

                # each entity
                for _entity in _entities:
                    if isinstance(_entity, str):
                        _entity_tokens = self.tokenizer.tokenize(_entity)
                        _entity_ids = self.tokenizer.convert_tokens_to_ids(_entity_tokens)
                    elif isinstance(_entity, list):
                        if isinstance(_entity[0], str):
                            _entity_ids = self.tokenizer.convert_tokens_to_ids(_y)
                        else:
                            raise ValueError("Wrong input format (line %d): \"%s\". " % (idx, _entity))

                    # search and tag
                    start_positions = com.find_all_boyer_moore(_input_ids, _entity_ids)
                    if not start_positions:
                        tf.logging.warning(
                            "Failed to find the mapping of entity "
                            "to inputs at line %d. A possible "
                            "reason is that the entity span is "
                            "truncated due to the "
                            "`max_seq_length` setting."
                            % (idx)
                        )
                        continue

                    for start_position in start_positions:
                        end_position = start_position + len(_entity) - 1
                        if start_position == end_position:
                            _label_ids[start_position] = type_B_id[_key] + 3
                        else:
                            for i in range(start_position, end_position + 1):
                                _label_ids[i] = type_B_id[_key] + 1
                            _label_ids[start_position] = type_B_id[_key]
                            _label_ids[end_position] = type_B_id[_key] + 2

            label_ids.append(_label_ids)
        if not self.entity_types:
            self.entity_types = list(type_B_id.keys())
        return label_ids

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            label_ids=split_placeholders["label_ids"],
            label_size=1 + len(self.entity_types) * 4,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_logits = output_arrays[0]
        batch_transition_matrix = output_arrays[1]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ""
        for key in metrics:
            info += ", %s %.4f" % (key, metrics[key])
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(_logits[:_input_length], transition_matrix)

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            _preds = {}
            for i, entity_type in enumerate(self.entity_types):
                _B_id = 1 + 4 * i
                _entities = self._get_entities(_viterbi_seq, _B_id)
                if _entities:
                    _preds[entity_type] = []

                    for _entity in _entities:
                        _start, _end = _entity[0], _entity[1]
                        if tokenized:
                            _entity_tokens = _tokens[_start: _end + 1]
                            _preds[entity_type].append(_entity_tokens)
                        else:
                            try:
                                _text_start = _mapping_start[_start]
                                _text_end = _mapping_end[_end]
                            except Exception:
                                continue
                            _entity_text = _text[_text_start: _text_end]
                            _preds[entity_type].append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs["preds"] = preds
        outputs["logits"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data["input_mask"]
        labels = self.data["label_ids"]
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        for key in metrics:
            outputs[key] = metrics[key]
        outputs["loss"] = loss

        return outputs


class BERTMRC(BERTClassifier, MRCModule):
    """ Machine reading comprehension on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=256,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._on_predict = False

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

    def predict(self, X=None, X_tokenized=None, batch_size=8):

        self._on_predict = True
        ret = super(MRCModule, self).predict(X, X_tokenized, batch_size)
        self._on_predict = False

        return ret

    predict.__doc__ = MRCModule.predict.__doc__

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, segment_ids, doc_ids, doc_text, doc_start) = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y, doc_ids, doc_text, doc_start, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception:
                raise ValueError(
                    "Wrong input format (line %d): \"%s\". "
                    "An untokenized example: "
                    "`X = [{\"doc\": \"...\", \"question\": \"...\", ...}, "
                    "...]`" % (idx, sample)
                )

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            _doc_tokens = segments.pop("doc")
            segments = list(segments.values()) + [_doc_tokens]
            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            _doc_tokens = segments[-1]

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))
            _doc_start = len(_input_tokens) - len(_doc_tokens) - 1

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _doc_ids = _input_ids[_doc_start: -1]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[idx]["doc"])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, segment_ids, doc_ids, doc_text, doc_start)

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
                    "Wrong input format of `y`. An untokenized example: "
                    "`y = [{\"text\": \"Obama\", \"answer_start\": 12}, "
                    "None, ...]`"
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

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, 2], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders["label_ids"],
            sample_weight=split_placeholders.get("sample_weight"),
            scope="mrc",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
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

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

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


class BERTVerifierMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on BERT, with a external front
    verifier. """
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=256,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        drop_pooler=False,
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._drop_pooler = drop_pooler
        self._on_predict = False

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

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, segment_ids, doc_ids, doc_text, doc_start) = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids, has_answer = self._convert_y(y, doc_ids, doc_text, doc_start, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)
            data["has_answer"] = np.array(has_answer, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []
        has_answer = []

        invalid_ids = []
        for idx, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                has_answer.append(0)
                continue

            if not isinstance(_y, dict) or "text" not in _y or "answer_start" not in _y:
                raise ValueError(
                    "Wrong input format of `y`. An untokenized example: "
                    "`y = [{\"text\": \"Obama\", \"answer_start\": 12}, "
                    "None, ...]`"
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

                start_positions = com.find_all_boyer_moore(_doc_ids, _answer_ids)
                if _overlap_time >= len(start_positions):
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(idx)
                    continue
                start_position = start_positions[_overlap_time]
                end_position = start_position + len(_answer_ids) - 1

            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning("Wrong `answer_start` at line %d. Ignored and set label to null text.")
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(idx)
                    continue

            label_ids.append([start_position + doc_start[idx], end_position + doc_start[idx]])
            has_answer.append(1)

        if invalid_ids:
            tf.logging.warning(
                "Failed to find the mapping of answer to inputs at "
                "line %s. A possible reason is that the answer spans "
                "are truncated due to the `max_seq_length` setting."
                % invalid_ids
            )

        return label_ids, has_answer

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, 2], tf.int32),
            "has_answer": com.get_placeholder(target, "has_answer", [None], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        verifier = ClsDecoder(
            is_training=is_training,
            input_tensor=encoder.get_pooled_output(),
            label_ids=split_placeholders["has_answer"],
            label_size=2,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/verifier",
            **kwargs,
        )
        if is_training:
            sample_weight = tf.cast(split_placeholders["has_answer"], tf.float32) * split_placeholders.get("sample_weight")
        else:
            sample_weight = split_placeholders.get("sample_weight")
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder.get_sequence_output(),
            label_ids=split_placeholders["label_ids"],
            sample_weight=sample_weight,
            scope="mrc",
            **kwargs,
        )

        verifier_total_loss, verifier_tensors = verifier.get_forward_outputs()
        decoder_total_loss, decoder_tensors = decoder.get_forward_outputs()

        total_loss = verifier_total_loss + decoder_total_loss
        tensors = collections.OrderedDict()
        for key in verifier_tensors:
            tensors["verifier_" + key] = verifier_tensors[key]
        for key in decoder_tensors:
            tensors["mrc_" + key] = decoder_tensors[key]

        return total_loss, tensors

    def _get_fit_ops(self, as_feature=False):
        ops = [
            self._tensors["verifier_preds"],
            self._tensors["verifier_losses"],
            self._tensors["mrc_preds"],
            self._tensors["mrc_losses"],
        ]
        if as_feature:
            ops.extend([self.placeholders["label_ids"]])
            ops.extend([self.placeholders["has_answer"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-2]
            batch_has_answer = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_ids"]]
            batch_has_answer = feed_dict[self.placeholders["has_answer"]]

        # verifier accuracy
        batch_has_answer_preds = output_arrays[0]
        has_answer_accuracy = np.mean(batch_has_answer_preds == batch_has_answer)

        # verifier loss
        batch_verifier_losses = output_arrays[1]
        verifier_loss = np.mean(batch_verifier_losses)

        # mrc exact match & f1
        batch_preds = output_arrays[2]
        for i in range(len(batch_has_answer_preds)):
            if batch_has_answer_preds[i] == 0:
                batch_preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # mrc loss
        batch_losses = output_arrays[3]
        loss = np.mean(batch_losses)

        info = ""
        info += ", has_ans_accuracy %.4f" % has_answer_accuracy
        info += ", exact_match %.4f" % exact_match
        info += ", f1 %.4f" % f1
        info += ", verifier_loss %.6f" % verifier_loss
        info += ", mrc_loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [
            self._tensors["verifier_probs"],
            self._tensors["verifier_preds"],
            self._tensors["mrc_probs"],
            self._tensors["mrc_preds"],
        ]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier preds & probs
        verifier_probs = com.transform(output_arrays[0], n_inputs)[:, 1]
        verifier_preds = com.transform(output_arrays[1], n_inputs)

        # mrc preds & probs
        probs = com.transform(output_arrays[2], n_inputs)
        mrc_preds = com.transform(output_arrays[3], n_inputs)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for idx, _preds in enumerate(mrc_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if verifier_preds[idx] == 0 or _start == 0 or _end == 0 or _start > _end:
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
        outputs["verifier_probs"] = verifier_probs
        outputs["verifier_preds"] = verifier_preds
        outputs["mrc_probs"] = probs
        outputs["mrc_preds"] = preds

        return outputs

    def _get_score_ops(self):
        return [
            self._tensors["verifier_preds"],
            self._tensors["verifier_losses"],
            self._tensors["mrc_preds"],
            self._tensors["mrc_losses"],
        ]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier accuracy
        has_answer_preds = com.transform(output_arrays[0], n_inputs)
        has_answer_accuracy = np.mean(has_answer_preds == self.data["has_answer"])

        # verifier loss
        verifier_losses = com.transform(output_arrays[1], n_inputs)
        verifier_loss = np.mean(verifier_losses)

        # mrc exact match & f1
        preds = com.transform(output_arrays[2], n_inputs)
        for i in range(len(has_answer_preds)):
            if has_answer_preds[i] == 0:
                preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(preds, self.data["label_ids"])

        # mrc loss
        losses = com.transform(output_arrays[3], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["has_ans_accuracy"] = has_answer_accuracy
        outputs["exact_match"] = exact_match
        outputs["f1"] = f1
        outputs["verifier_loss"] = verifier_loss
        outputs["mrc_loss"] = loss

        return outputs


class BERTLM(LMModule):
    """ Language modeling on BERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
        do_sample_next_sentence=True,
        max_predictions_per_seq=20,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        do_whole_word_mask=False,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(BERTLM, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = 2
        self.do_sample_next_sentence = do_sample_next_sentence
        self.masked_lm_prob = masked_lm_prob
        self.short_seq_prob = short_seq_prob
        self.do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._max_predictions_per_seq = max_predictions_per_seq
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
            if y is not None:
                assert not self.do_sample_next_sentence, "`y` should be None when `do_sample_next_sentence` is True."
            else:
                assert self.do_sample_next_sentence, "`y` can't be None when `do_sample_next_sentence` is False."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights,
             next_sentence_labels) = self._convert_X(X_tokenized if tokenized else X, is_training, tokenized=tokenized)

            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["masked_lm_positions"] = np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data["masked_lm_ids"] = np.array(masked_lm_ids, dtype=np.int32)
                data["masked_lm_weights"] = np.array(masked_lm_weights, dtype=np.float32)

            if is_training and self.do_sample_next_sentence:
                data["next_sentence_labels"] = np.array(next_sentence_labels, dtype=np.int32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            next_sentence_labels = self._convert_y(y)
            data["next_sentence_labels"] = np.array(next_sentence_labels, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception:
                tf.logging.warning("Wrong input format (line %d): \"%s\". " % (idx, sample))

        input_ids = []
        input_mask = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_labels = []

        # random sampling of next sentence
        if is_training and self.do_sample_next_sentence:
            new_segment_input_tokens = []
            for idx in range(len(segment_input_tokens)):
                instances = create_instances_from_document(
                    all_documents=segment_input_tokens,
                    document_index=idx,
                    max_seq_length=self.max_seq_length - 3,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    short_seq_prob=self.short_seq_prob,
                    vocab_words=list(self.tokenizer.vocab.keys()),
                )
                for (segments, is_random_next) in instances:
                    new_segment_input_tokens.append(segments)
                    next_sentence_labels.append(is_random_next)
            segment_input_tokens = new_segment_input_tokens

        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _masked_lm_positions = []
            _masked_lm_ids = []
            _masked_lm_weights = []

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            # random sampling of masked tokens
            if is_training:
                if (idx + 1) % 10000 == 0:
                    tf.logging.info("Sampling masks of input %d" % (idx + 1))
                _input_tokens, _masked_lm_positions, _masked_lm_labels = create_masked_lm_predictions(
                    tokens=_input_tokens,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    vocab_words=list(self.tokenizer.vocab.keys()),
                    do_whole_word_mask=self.do_whole_word_mask,
                )
                _masked_lm_ids = self.tokenizer.convert_tokens_to_ids(_masked_lm_labels)
                _masked_lm_weights = [1.0] * len(_masked_lm_positions)

                # padding
                for _ in range(self._max_predictions_per_seq - len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)
                    _masked_lm_ids.append(0)
                    _masked_lm_weights.append(0.0)
            else:
                # `masked_lm_positions` is required for both training
                # and inference of BERT language modeling.
                for i in range(len(_input_tokens)):
                    if _input_tokens[i] == "[MASK]":
                        _masked_lm_positions.append(i)

                # padding
                for _ in range(self._max_predictions_per_seq - len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels)

    def _convert_x(self, x, tokenized):
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
        label_set = set(y)

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, "Number of unique `y`s exceeds 2."
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

        label_ids = [self._label_to_id[label] for label in y]
        return label_ids

    def _set_placeholders(self, target, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "masked_lm_positions": com.get_placeholder(target, "masked_lm_positions", [None, self._max_predictions_per_seq], tf.int32),
            "masked_lm_ids": com.get_placeholder(target, "masked_lm_ids", [None, self._max_predictions_per_seq], tf.int32),
            "masked_lm_weights": com.get_placeholder(target, "masked_lm_weights", [None, self._max_predictions_per_seq], tf.float32),
            "next_sentence_labels": com.get_placeholder(target, "next_sentence_labels", [None], tf.int32),
            "sample_weight": com.get_placeholder(target, "sample_weight", [None], tf.float32),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        decoder = BERTDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            encoder=encoder,
            masked_lm_positions=split_placeholders["masked_lm_positions"],
            masked_lm_ids=split_placeholders["masked_lm_ids"],
            masked_lm_weights=split_placeholders["masked_lm_weights"],
            next_sentence_labels=split_placeholders["next_sentence_labels"],
            sample_weight=split_placeholders.get("sample_weight"),
            scope_lm="cls/predictions",
            scope_cls="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [
            self._tensors["MLM_preds"],
            self._tensors["NSP_preds"],
            self._tensors["MLM_losses"],
            self._tensors["NSP_losses"],
        ]
        if as_feature:
            ops.extend([
                self.placeholders["masked_lm_positions"],
                self.placeholders["masked_lm_ids"],
                self.placeholders["next_sentence_labels"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mlm_positions = output_arrays[-3]
            batch_mlm_labels = output_arrays[-2]
            batch_nsp_labels = output_arrays[-1]
        else:
            batch_mlm_positions = feed_dict[self.placeholders["masked_lm_positions"]]
            batch_mlm_labels = feed_dict[self.placeholders["masked_lm_ids"]]
            batch_nsp_labels = feed_dict[self.placeholders["next_sentence_labels"]]

        # MLM accuracy
        batch_mlm_preds = output_arrays[0]
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) / batch_mlm_mask.sum()

        # NSP accuracy
        batch_nsp_preds = output_arrays[1]
        nsp_accuracy = np.mean(batch_nsp_preds == batch_nsp_labels)

        # MLM loss
        batch_mlm_losses = output_arrays[2]
        mlm_loss = np.mean(batch_mlm_losses)

        # NSP loss
        batch_nsp_losses = output_arrays[3]
        nsp_loss = np.mean(batch_nsp_losses)

        info = ""
        info += ", MLM accuracy %.4f" % mlm_accuracy
        info += ", NSP accuracy %.4f" % nsp_accuracy
        info += ", MLM loss %.6f" % mlm_loss
        info += ", NSP loss %.6f" % nsp_loss

        return info

    def _get_predict_ops(self):
        return [
            self._tensors["MLM_preds"],
            self._tensors["NSP_preds"],
            self._tensors["NSP_probs"],
        ]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # MLM preds
        mlm_preds = []
        mlm_positions = self.data["masked_lm_positions"]
        all_preds = com.transform(output_arrays[0], n_inputs).tolist()
        for idx, _preds in enumerate(all_preds):
            _ids = []
            for p_id, _id in enumerate(_preds):
                if mlm_positions[idx][p_id] == 0:
                    break
                _ids.append(_id)
            mlm_preds.append(self.tokenizer.convert_ids_to_tokens(_ids))

        # NSP preds
        nsp_preds = com.transform(output_arrays[1], n_inputs).tolist()

        # NSP probs
        nsp_probs = com.transform(output_arrays[2], n_inputs)

        outputs = {}
        outputs["mlm_preds"] = mlm_preds
        outputs["nsp_preds"] = nsp_preds
        outputs["nsp_probs"] = nsp_probs

        return outputs
