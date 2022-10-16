import numpy as np

from .base import ClassifierModule
from .bert import BERTClassifier
from ..model.base import ClsDecoder
from ..model.textcnn import TextCNNEncoder, get_decay_power
from ..token import WordPieceTokenizer
from ..third import tf
from .. import com


class TextCNNClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on TextCNN. """
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

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

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._filter_sizes = filter_sizes
        self._num_channels = num_channels
        self._hidden_size = hidden_size
        self._id_to_label = None

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
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, _, _ = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
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

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = TextCNNEncoder(
            vocab_size=len(self.tokenizer.vocab),
            filter_sizes=self._filter_sizes,
            num_channels=self._num_channels,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            embedding_size=self._hidden_size,
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
