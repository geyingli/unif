from .._base_._base_mrc import MRCModule
from ..bert.bert_mrc import BERTMRC
from .._base_._base_ import MRCDecoder
from ..bert.bert import BERTEncoder, BERTConfig, get_decay_power
from ...token import WordPieceTokenizer
from ...third import tf


class ELECTRAMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on ELECTRA. """
    
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
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
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
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=placeholders["label_ids"],
            sample_weight=placeholders.get("sample_weight"),
            scope="mrc",
            **kwargs,
        )
        return decoder.get_forward_outputs()
