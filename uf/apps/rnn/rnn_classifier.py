import numpy as np

from .rnn import RNNEncoder, get_decay_power
from .._base_._base_classifier import ClsDecoder, ClassifierModule
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class RNNClassifier(ClassifierModule):
    """ Single-label classifier on RNN/LSTM/GRU. """

    def __init__(
        self,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        rnn_core="lstm",
        hidden_size=128,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._rnn_core = rnn_core
        self._hidden_size = hidden_size

        assert rnn_core in ("rnn", "lstm", "gru"), (f"Invalid `rnn_core`: {rnn_core}. Pick one from \"rnn\", \"lstm\" and \"gru\".")
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power()

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
            input_ids, seq_length = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["seq_length"] = np.array(seq_length, dtype=np.int32)
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
        seq_length = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = []
            _input_ids = []
            _seq_length = 0

            com.truncate_segments(segments, self.max_seq_length - len(segments), truncate_method=self.truncate_method)
            for segment in segments:
                _input_tokens.extend(segment + ["[SEP]"])

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _seq_length = len(_input_ids)

            # padding
            _input_ids += [0] * (self.max_seq_length - len(_input_ids))

            input_ids.append(_input_ids)
            seq_length.append(_seq_length)

        return input_ids, seq_length

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "seq_length": tf.placeholder(tf.int32, [None], "seq_length"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = RNNEncoder(
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            seq_length=placeholders["seq_length"],
            vocab_size=len(self.tokenizer.vocab),
            rnn_core=self._rnn_core,
            hidden_size=self._hidden_size,
            scope=self._rnn_core,
            trainable=True,
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
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors
