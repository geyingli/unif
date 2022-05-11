from ..thirdparty import tf
from .base import ClassifierModule
from .bert import BERTClassifier, get_bert_config, get_key_to_depths
from ..modeling.performer import PerformerEncoder
from ..modeling.base import CLSDecoder
from ..tokenization import WordPieceTokenizer


class PerformerClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on Performer. """
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 kernel_transformation="relu",
                 nb_random_features=1,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method="LIFO"):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._id_to_label = None
        self._kernel_transformation = kernel_transformation
        self._nb_random_features = nb_random_features
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.bert_config.num_hidden_layers)

        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            self.bert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            self.bert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = PerformerEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            kernel_transformation=self._kernel_transformation,
            nb_random_features=self._nb_random_features,
            drop_pooler=self._drop_pooler,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = CLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            **kwargs)
        return decoder.get_forward_outputs()
