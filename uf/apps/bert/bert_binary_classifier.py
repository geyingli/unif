from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_classifier import BERTClassifier
from ..base.base_classifier import ClassifierModule
from ..base.base import BinaryClsDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


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

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, self.label_size], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()
        decoder = BinaryClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            label_weight=self.label_weight,
            scope="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_predict_ops(self):
        return [self._tensors["probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

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