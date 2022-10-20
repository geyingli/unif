from .._base_._base_seq_classifier import SeqClsDecoder, SeqClassifierModule
from ..bert.bert_seq_classifier import BERTSeqClassifier
from ..bert.bert import BERTEncoder, BERTConfig, get_decay_power
from ...token import WordPieceTokenizer
from ...third import tf


class ELECTRASeqClassifier(BERTSeqClassifier, SeqClassifierModule):
    """ Sequence labeling classifier on ELECTRA. """

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
        super(SeqClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method

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

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            scope="electra",
            drop_pooler=True,
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
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors
