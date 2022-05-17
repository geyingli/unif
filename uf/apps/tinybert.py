import os
import copy
import numpy as np

from .base import ClassifierModule
from .bert import BERTClassifier, BERTBinaryClassifier
from ..model.bert import BERTConfig
from ..model.tinybert import TinyBERTCLSDistillor, TinyBERTBinaryCLSDistillor
from ..token import WordPieceTokenizer
from ..third import tf


class TinyBERTClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on TinyBERT, a distillation model. """
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

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
        hidden_size=384,
        num_hidden_layers=4,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._id_to_label = None

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = "unsupported"

        self.student_config = copy.deepcopy(self.bert_config)
        self.student_config.hidden_size = hidden_size
        self.student_config.intermediate_size = 4 * hidden_size
        self.student_config.num_hidden_layers = num_hidden_layers

        assert label_size, ("`label_size` can't be None.")
        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            self.bert_config.vocab_size += 1
            self.student_config.vocab_size += 1
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            self.bert_config.vocab_size += 1
            self.student_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def to_bert(self, save_dir):
        """ Isolate student tiny_bert out of traing graph. """
        if not self._session_built:
            raise ValueError("Init, fit, predict or score before saving checkpoint.")

        tf.gfile.MakeDirs(save_dir)

        tf.logging.info("Saving checkpoint into %s/bert_model.ckpt" % save_dir)
        self.init_checkpoint = save_dir + "/bert_model.ckpt"

        assignment_map = {}
        for var in self.global_variables:
            if var.name.startswith("tiny/"):
                assignment_map[var.name.replace("tiny/", "")[:-2]] = var
        saver = tf.train.Saver(assignment_map, max_to_keep=1000000)
        saver.save(self.sess, self.init_checkpoint)

        self.student_config.to_json_file(os.path.join(save_dir, "bert_config.json"))

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = TinyBERTCLSDistillor(
            student_config=self.student_config,
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            label_ids=split_placeholders.get("label_ids"),
            sample_weight=split_placeholders.get("sample_weight"),
            drop_pooler=self._drop_pooler,
            label_size=self.label_size,
            **kwargs,
        )
        return model.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        return [self._tensors["losses"]]

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        # loss
        batch_losses = output_arrays[0]
        loss = np.mean(batch_losses)

        info = ""
        info += ", distill loss %.6f" % loss

        return info


class TinyBERTBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    """ Multi-label classifier on TinyBERT, a distillation model. """
    _INFER_ATTRIBUTES = BERTBinaryClassifier._INFER_ATTRIBUTES

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
        hidden_size=384,
        num_hidden_layers=4,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._id_to_label = None

        self.bert_config = BERTConfig.from_json_file(config_file)
        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = "unsupported"

        self.student_config = copy.deepcopy(self.bert_config)
        self.student_config.hidden_size = hidden_size
        self.student_config.intermediate_size = 4 * hidden_size
        self.student_config.num_hidden_layers = num_hidden_layers

        assert label_size, ("`label_size` can't be None.")
        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            self.bert_config.vocab_size += 1
            self.student_config.vocab_size += 1
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            self.bert_config.vocab_size += 1
            self.student_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def to_bert(self, save_dir):
        """ Isolate student tiny_bert out of traing graph. """
        if not self._session_built:
            raise ValueError("Init, fit, predict or score before saving checkpoint.")

        tf.gfile.MakeDirs(save_dir)

        tf.logging.info("Saving checkpoint into %s/bert_model.ckpt" % (save_dir))
        self.init_checkpoint = (
            save_dir + "/bert_model.ckpt")

        assignment_map = {}
        for var in self.global_variables:
            if var.name.startswith("tiny/"):
                assignment_map[var.name.replace("tiny/", "")[:-2]] = var
        saver = tf.train.Saver(assignment_map, max_to_keep=1000000)
        saver.save(self.sess, self.init_checkpoint)

        self.student_config.to_json_file(os.path.join(save_dir, "bert_config.json"))

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        if y:
            # convert y and sample_weight
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = TinyBERTBinaryCLSDistillor(
            student_config=self.student_config,
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            label_ids=split_placeholders.get("label_ids"),
            sample_weight=split_placeholders.get("sample_weight"),
            drop_pooler=self._drop_pooler,
            label_size=self.label_size,
            **kwargs,
        )
        return model.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        return [self._tensors["losses"]]

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        # loss
        batch_losses = output_arrays[0]
        loss = np.mean(batch_losses)

        info = ""
        info += ", distill loss %.6f" % loss

        return info
