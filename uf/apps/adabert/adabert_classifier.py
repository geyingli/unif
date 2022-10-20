import numpy as np

from .adabert import AdaBERTClsDistillor
from .._base_._base_classifier import ClassifierModule
from ..bert.bert_classifier import BERTClassifier
from ..bert.bert import BERTConfig
from ...token import WordPieceTokenizer
from ...third import tf


class AdaBERTClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on AdaBERT, a distillation model. """

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
        k_max=4,
        num_intermediates=3,
        embedding_size=128,
        temp_decay_steps=18000,
        model_l2_reg=3e-4,
        arch_l2_reg=1e-3,
        loss_gamma=0.8,
        loss_beta=4.0,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._k_max = k_max
        self._num_intermediates = num_intermediates
        self._embedding_size = embedding_size
        self._temp_decay_steps = temp_decay_steps
        self._model_l2_reg = model_l2_reg
        self._arch_l2_reg = arch_l2_reg
        self._loss_gamma = loss_gamma
        self._loss_beta = loss_beta

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = "unsupported"

        assert label_size, ("`label_size` can't be None.")
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

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y is not None:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _forward(self, is_training, placeholders, **kwargs):

        model = AdaBERTClsDistillor(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            label_ids=placeholders.get("label_ids"),
            sample_weight=placeholders.get("sample_weight"),
            drop_pooler=self._drop_pooler,
            label_size=self.label_size,
            k_max=self._k_max,
            num_intermediates=self._num_intermediates,
            embedding_size=self._embedding_size ,
            temp_decay_steps=self._temp_decay_steps,
            model_l2_reg=self._model_l2_reg,
            arch_l2_reg=self._arch_l2_reg,
            loss_gamma=self._loss_gamma,
            loss_beta=self._loss_beta,
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        return [self.tensors["losses"]]

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        # loss
        batch_losses = output_arrays[0]
        loss = np.mean(batch_losses)

        info = ""
        info += ", distill loss %.6f" % loss

        return info
