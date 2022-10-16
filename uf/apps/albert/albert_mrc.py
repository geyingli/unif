from .albert import ALBERTEncoder, ALBERTConfig, get_decay_power
from ..base.base_mrc import MRCModule
from ..bert.bert_mrc import BERTMRC
from ..base.base import MRCDecoder
from ...token import WordPieceTokenizer
from ...third import tf


class ALBERTMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on ALBERT. """
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
        self._on_predict = False
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

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            albert_config=self.albert_config,
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
