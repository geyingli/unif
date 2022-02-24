""" Applications based on DilatedBERT. """

import random
import numpy as np

from ..thirdparty import tf
from .base import LMModule
from .bert import get_bert_config, get_key_to_depths
from ..modeling.dilated import DLM
from ..tokenization import WordPieceTokenizer
from .. import common


class DilatedLM(LMModule):
    """ Language modeling on DilatedBERT. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 replace_prob=0.05,
                 add_prob=0.05,
                 subtract_prob=0.05,
                 do_lower_case=True,
                 truncate_method="LIFO"):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._replace_prob = replace_prob
        self._add_prob = add_prob
        self._subtract_prob = subtract_prob
        self._loop = 1
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
        if "[SPAD]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SPAD]")
            self.bert_config.vocab_size += 1
            tf.logging.info("Add necessary token `[SPAD]` into vocabulary.")

    def predict(self, X=None, X_tokenized=None,
                batch_size=8, loop=1):
        """ Inference on the model.
        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            loop: int. Number of inference loop to rewrite the input.
        Returns:
            A dict object of model outputs.
        """

        if loop != self._loop:
            self._loop = loop
            self._session_mode = None

        return super(LMModule, self).predict(
            X, X_tokenized, batch_size)

    def export(self, export_dir, loop=1):
        """ Export model into SavedModel files.
        Args:
            export_dir: str. Directory to which the model is saved.
            loop: int. Number of inference loop to rewrite the input.
        Returns:
            None
        """

        if loop != self._loop:
            self._loop = loop
            self._session_mode = None

        return super(LMModule, self).export(export_dir)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, ("%s is unsupervised. `y` should be None."
                           % self.__class__.__name__)

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            (dilated_ids, label_ids) = self._convert_X(
                X_tokenized if tokenized else X, tokenized=tokenized,
                is_training=is_training)
            data["dilated_ids"] = np.array(dilated_ids, dtype=np.int32)

            if is_training:
                data["label_ids"] = np.array(label_ids, dtype=np.int32)

            n_inputs = len(dilated_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized, is_training):
        dilated_ids = []
        label_ids = []

        for ex_id, example in enumerate(X_target):
            try:
                _input_tokens = self._convert_x(example, tokenized)
            except Exception:
                raise ValueError(
                    "Wrong input format (line %d): \"%s\". "
                    % (ex_id, example))

            _input_tokens = ["[CLS]"] + _input_tokens
            _input_ids = self.tokenizer.convert_tokens_to_ids(
                _input_tokens)

            common.truncate_segments(
                [_input_ids], self.max_seq_length,
                truncate_method=self.truncate_method)
            nonpad_seq_length = len(_input_ids)
            _input_mask = [1] * nonpad_seq_length

            if nonpad_seq_length < self.max_seq_length:
                _input_ids.extend(
                    [0] * (self.max_seq_length - nonpad_seq_length))
                _input_mask.extend(
                    [0] * (self.max_seq_length - nonpad_seq_length))

            _dilated_ids = []
            _label_ids = []
            for i, _input_id in enumerate(_input_ids):
                _dilated_ids.extend([_input_id, 0])
                _label_ids.extend([_input_id, 0])

            # replace/add/subtract
            if is_training:
                max_replace = int(nonpad_seq_length * self._replace_prob)
                max_add = int(nonpad_seq_length * self._add_prob)
                max_subtract = int(nonpad_seq_length * self._subtract_prob)

                sample_wrong_tokens(
                    _dilated_ids, _label_ids,
                    max_replace, max_add, max_subtract,
                    nonpad_seq_length=nonpad_seq_length,
                    vocab_size=len(self.tokenizer.vocab))

            dilated_ids.append(_dilated_ids)
            label_ids.append(_label_ids)

        return dilated_ids, label_ids

    def _convert_x(self, x, tokenized):
        if not tokenized:
            # deal with general inputs
            if isinstance(x, str):
                return self.tokenizer.tokenize(x)

        # deal with tokenized inputs
        elif isinstance(x[0], str):
            return x

        # deal with tokenized and multiple inputs
        raise ValueError(
            "%s only supports single sentence inputs."
            % self.__class__.__name__)

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            "dilated_ids": common.get_placeholder(
                target, "dilated_ids",
                [None, self.max_seq_length * 2], tf.int32),
            "label_ids": common.get_placeholder(
                target, "label_ids",
                [None, self.max_seq_length * 2], tf.int32),
        }
        if not on_export:
            self.placeholders["sample_weight"] = \
                common.get_placeholder(
                    target, "sample_weight",
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        model = DLM(
            bert_config=self.bert_config,
            is_training=is_training,
            dilated_ids=split_placeholders["dilated_ids"],
            label_ids=split_placeholders["label_ids"],
            max_seq_length=self.max_seq_length,
            spad_id=self.tokenizer.convert_tokens_to_ids(["[SPAD]"])[0],
            loop=self._loop,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="dilated",
            **kwargs)
        return model.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["LM"], self._tensors["LM"]]
        if as_feature:
            ops.extend([self.placeholders["dilated_ids"],
                        self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_inputs = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_inputs = feed_dict[self.placeholders["dilated_ids"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # accuracy
        batch_preds = output_arrays[0]
        batch_mask = (batch_inputs != batch_labels)
        accuracy = np.sum((batch_preds == batch_labels) * batch_mask) / \
            (np.sum(batch_mask) + 1e-6)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", accuracy %.4f" % accuracy
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["LM"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        all_preds = common.transform(output_arrays[0], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_ids = [_pred_id for _pred_id in _pred_ids[1:]
                         if _pred_id != 0]
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            _pred_text = common.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs["preds"] = preds

        return outputs


def sample_wrong_tokens(_dilated_ids, _label_ids,
                        max_replace, max_add, max_subtract,
                        nonpad_seq_length, vocab_size):

    # The sampling follows the order `add -> replace -> subtract`

    # `add`, remove padding for prediction of adding tokens
    # e.g. 124 0 591 0 9521 -> 124 591 9521 0 0
    for _ in range(max_add):
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] != 0 and
                         _dilated_ids[i - 1] == 0 and
                         _dilated_ids[i + 1] == 0]
        if not cand_indicies:
            break

        def mod_add(list_obj, index):
            list_obj.pop(index + 1)
            list_obj.pop(index - 1)
            list_obj.extend([0, 0])
        index = random.choice(cand_indicies)
        mod_add(_dilated_ids, index)
        mod_add(_label_ids, index)
        _dilated_ids[index - 1] = 0

    # `replace`, replace tokens for prediction of replacing tokens
    # e.g. 124 0 591 0 9521 -> 124 0 789 0 9521
    for _ in range(max_replace):
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] != 0 and
                         _dilated_ids[i - 1] == 0 and
                         _dilated_ids[i + 1] == 0 and
                         _dilated_ids[i] == _label_ids[i]]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _dilated_ids[index] = random.randint(1, vocab_size - 1)

    # `subtract`, add wrong tokens for prediction of subtraction
    # e.g. 124 0 591 0 9521 -> 124 0 92 0 591
    for _ in range(max_subtract):
        if _dilated_ids[-2] != 0:  # no more space
            break
        cand_indicies = [i for i in range(1, len(_dilated_ids) - 1)
                         if _dilated_ids[i] == 0 and
                         _dilated_ids[i - 1] != 0 and
                         _dilated_ids[i + 1] != 0 and
                         _dilated_ids[i - 1] == _label_ids[i - 1] and
                         _dilated_ids[i + 1] == _label_ids[i + 1]]
        if not cand_indicies:
            break

        index = random.choice(cand_indicies)
        _dilated_ids.insert(index, random.randint(1, vocab_size - 1))
        _dilated_ids.insert(index, 0)
        _dilated_ids.pop()
        _dilated_ids.pop()
        _label_ids.insert(index, 0)
        _label_ids.insert(index, 0)
        _label_ids.pop()
        _label_ids.pop()
