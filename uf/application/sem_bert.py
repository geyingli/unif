""" Applications based on SemBERT model. """

import copy
import numpy as np

from ..tools import tf
from .base import ClassifierModule
from .bert import BERTClassifier, get_bert_config
from ..modeling.bert import BERTEncoder
from ..modeling.sem_bert import SemBERTDecoder
from ..tokenization.word_piece import get_word_piece_tokenizer
from .. import utils


class SemBERTClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on SemBERT. """
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 sem_features=None,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method="LIFO"):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self.sem_features = sem_features
        self._drop_pooler = drop_pooler
        self._id_to_label = None
        self.__init_args__ = locals()

        self.bert_config = get_bert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
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

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can\"t be None."
        if is_parallel:
            assert self.label_size, (
                "Can\"t parse data on multi-processing "
                "when `label_size` is None.")

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            assert tokenized, (
                "Inputs of `%s` must be already tokenized "
                "and fed into `X_tokenized`." % self.__class__.__name__)
            (input_ids, input_mask, segment_ids,
             sem_features) = self._convert_X(
                 X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["sem_features"] = np.array(sem_features, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_inputs = []
        for ex_id, example in enumerate(X_target):
            try:
                assert len(example["Text"]) == len(example["Sem"])
                if isinstance(example["Text"][0], list):
                    for i in range(len(example["Text"])):
                        assert len(example["Text"][i]) == len(example["Sem"][i])
                sem = copy.deepcopy(example["Sem"])
                if not isinstance(sem[0], list):
                    sem = [sem]
                segment_inputs.append(
                    {"Sem": sem,
                     "Text": self._convert_x(example["Text"], tokenized)})
            except Exception:
                raise ValueError(
                    "Wrong input format (line %d): %s. An example: "
                    "X_tokenized = [{\"Sem\": [\"n\", \"v\", \"n\"], "
                    "\"Text\": [\"I\", \"love\", \"you\"]}, ...]"
                    % (ex_id, example))

        if self.sem_features is None:
            self.sem_features = set()
            for segments in segment_inputs:
                for segment in segments["Sem"]:
                    for feature in segment:
                        self.sem_features.add(feature)
            self.sem_features = list(self.sem_features)
        elif not isinstance(self.sem_features, list):
            raise ValueError(
                "`sem_features` should be a list of possible values "
                "(integer or string). E.g. [\"n\", \"v\", \"adj\"].")
        sem_features_map = {
            self.sem_features[i]: i + 3
            for i in range(len(self.sem_features))}

        input_ids = []
        input_mask = []
        segment_ids = []
        sem_features = []
        for ex_id, segments in enumerate(segment_inputs):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _sem_features = [1]  # same as [CLS]

            utils.truncate_segments(
                segments["Text"], self.max_seq_length - len(segments["Text"]) - 1,
                truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments["Text"]):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            for i in range(len(segments["Sem"])):
                segment = segments["Sem"][i]
                n = len(segments["Text"][i])
                for feature in segment[:n]:
                    try:
                        _sem_features.append(sem_features_map[feature])
                    except Exception:
                        tf.logging.warning(
                            "Unregistered semantic feature: %s. Ignored."
                            % feature)
                        continue
                _sem_features.append(2)  # same as [SEP]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)
                _sem_features.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            sem_features.append(_sem_features)

        return (input_ids, input_mask, segment_ids, sem_features)

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            "input_ids": utils.get_placeholder(
                target, "input_ids",
                [None, self.max_seq_length], tf.int32),
            "input_mask": utils.get_placeholder(
                target, "input_mask",
                [None, self.max_seq_length], tf.int32),
            "segment_ids": utils.get_placeholder(
                target, "segment_ids",
                [None, self.max_seq_length], tf.int32),
            "sem_features": utils.get_placeholder(
                target, "sem_features",
                [None, self.max_seq_length], tf.int32),
            "label_ids": utils.get_placeholder(
                target, "label_ids", [None], tf.int32),
        }
        if not on_export:
            self.placeholders["sample_weight"] = \
                utils.get_placeholder(
                    target, "sample_weight",
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            scope="bert",
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = SemBERTDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders["input_mask"],
            sem_features=split_placeholders["sem_features"],
            label_ids=split_placeholders["label_ids"],
            max_seq_length=self.max_seq_length,
            feature_size=len(self.sem_features),
            label_size=self.label_size,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            **kwargs)
        return decoder.get_forward_outputs()


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        "/embeddings": num_hidden_layers + 2,
        "sem/": 2,
        "/pooler/": 1,
        "cls/": 0}
    for layer_idx in range(num_hidden_layers):
        key_to_depths["/layer_%d/" % layer_idx] = \
            num_hidden_layers - layer_idx + 1
    return key_to_depths
