import numpy as np

from .textcnn import TextCNNEncoder, get_decay_power
from .._base_._base_classifier import ClassifierModule
from .._base_._base_ import ClsDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from .. import com


class TextCNNClassifier(ClassifierModule):
    """ Single-label classifier on TextCNN. """

    def __init__(
        self,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        filter_sizes="2,4,6",
        num_channels=6,
        hidden_size=256,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._filter_sizes = filter_sizes
        self._num_channels = num_channels
        self._hidden_size = hidden_size

        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power()

        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
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
            input_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
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
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))

        input_ids = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _input_tokens.extend(segment + ["[SEP]"])

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)

            input_ids.append(_input_ids)

        return input_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = TextCNNEncoder(
            vocab_size=len(self.tokenizer.vocab),
            filter_sizes=self._filter_sizes,
            num_channels=self._num_channels,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            embedding_size=self._hidden_size,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()
        decoder = ClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()
