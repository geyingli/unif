from .albert import ALBERTEncoder, ALBERTConfig,  get_decay_power
from .._base_._base_classifier import ClassifierModule
from ..bert.bert_seq_classifier import BERTSeqClassifier
from .._base_._base_ import SeqClsDecoder
from ...token import WordPieceTokenizer
from ...third import tf


class ALBERTSeqClassifier(BERTSeqClassifier, ClassifierModule):
    """ Sequence labeling classifier on ALBERT. """
    
    _INFER_ATTRIBUTES = BERTSeqClassifier._INFER_ATTRIBUTES

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

        self.albert_config = ALBERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(self.albert_config.num_hidden_layers)

        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            self.albert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            self.albert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = ALBERTEncoder(
            albert_config=self.albert_config,
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
