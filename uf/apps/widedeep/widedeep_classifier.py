import numpy as np

from .widedeep import WideDeepClsDecoder, get_decay_power
from .._base_._base_classifier import ClassifierModule
from ..bert.bert import BERTEncoder, BERTConfig
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class WideDeepClassifier(ClassifierModule):
    """ Single-label classifier on Wide & Deep model with BERT. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
        "wide_features": "A list of possible values for `Wide` features (integer or string)",
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
        wide_features=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self.wide_features = wide_features

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
            (input_ids, input_mask, segment_ids,
             n_wide_features, wide_features) = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["n_wide_features"] = np.array(n_wide_features, dtype=np.int32)
            data["wide_features"] = np.array(wide_features, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y is not None:
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_inputs = []
        for idx, sample in enumerate(X_target):
            try:
                segment_inputs.append({
                    "Wide": sample["Wide"],
                    "Deep": self._convert_x(sample["Deep"], tokenized),
                })
            except Exception as e:
                raise ValueError(
                    "Wrong input format (%s): %s. An untokenized "
                    "example: X = [{\"Wide\": [1, 5, \"positive\"], "
                    "\"Deep\": \"I bet she will win.\"}, ...]"
                    % (sample, e)
                )

        if self.wide_features is None:
            self.wide_features = set()
            for segments in segment_inputs:
                for feature in segments["Wide"]:
                    self.wide_features.add(feature)
            self.wide_features = list(self.wide_features)
        elif not isinstance(self.wide_features, list):
            raise ValueError(
                "`wide_features` should be a list of possible values "
                "(integer or string). "
                "E.g. [1, \"Positive\", \"Subjective\"]."
            )
        wide_features_map = {self.wide_features[i]: i + 1 for i in range(len(self.wide_features))}

        input_ids = []
        input_mask = []
        segment_ids = []
        n_wide_features = []
        wide_features = []
        for idx, segments in enumerate(segment_inputs):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _wide_features = []
            for feature in segments["Wide"]:
                try:
                    _wide_features.append(wide_features_map[feature])
                except Exception:
                    tf.logging.warning("Unregistered wide feature: %s. Ignored." % feature)
                    continue
            _n_wide_features = len(_wide_features)

            segments = segments["Deep"]
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
            for _ in range(len(self.wide_features) - _n_wide_features):
                _wide_features.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            n_wide_features.append(_n_wide_features)
            wide_features.append(_wide_features)

        return input_ids, input_mask, segment_ids, n_wide_features, wide_features

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "n_wide_features": tf.placeholder(tf.int32, [None], "n_wide_features"),
            "wide_features": tf.placeholder(tf.int32, [None, len(self.wide_features)], "wide_features"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
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
        encoder_output = encoder.get_pooled_output()
        decoder = WideDeepClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            n_wide_features=placeholders["n_wide_features"],
            wide_features=placeholders["wide_features"],
            label_ids=placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

