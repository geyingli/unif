import numpy as np

from .uda import UDADecoder
from .._base_._base_classifier import ClassifierModule
from ..bert.bert import BERTEncoder, BERTConfig, get_decay_power
from ..bert.bert_classifier import BERTClassifier
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com
from .. import util


class UDAClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on UDA. """

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
        uda_softmax_temp=-1,
        uda_confidence_thresh=-1,
        tsa_schedule="linear",
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._uda_softmax_temp = uda_softmax_temp
        self._uda_confidence_thresh = uda_confidence_thresh
        self._tsa_schedule = tsa_schedule

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

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        # simplified when not training
        if not is_training:
            return super().convert(X, y, sample_weight, X_tokenized, is_training)

        if is_training:
            assert y is not None, "`y` can't be None."
        if is_parallel:
            assert self.label_size, "Can't parse data on multi-processing when `label_size` is None."

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            (input_ids, input_mask, segment_ids,
             aug_input_ids, aug_input_mask, aug_segment_ids,
             is_supervised) = self._convert_X_reimp(X_tokenized if tokenized else X, y, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["aug_input_ids"] = np.array(aug_input_ids, dtype=np.int32)
            data["aug_input_mask"] = np.array(aug_input_mask, dtype=np.int32)
            data["aug_segment_ids"] = np.array(aug_segment_ids, dtype=np.int32)
            data["is_supervised"] = np.array(is_supervised, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y is not None:
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X_reimp(self, X_target, y, tokenized):

        # tokenize input texts
        sup_ori_input_tokens = []
        aug_input_tokens = []
        is_supervised = []
        for idx, sample in enumerate(X_target):
            try:
                label = y[idx]

                if label is None:
                    assert len(sample) == 2
                    sup_ori_input_tokens.append(self._convert_x(sample[0], tokenized))
                    aug_input_tokens.append(self._convert_x(sample[1], tokenized))
                    is_supervised.append(0)
                else:
                    sup_ori_input_tokens.append(self._convert_x(sample, tokenized))
                    aug_input_tokens.append([])
                    is_supervised.append(1)
            except AssertionError:
                assert False, "Must have exactly two sentence input for an unsupervised example, respectively original and augmented."
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))

        input_ids = []
        input_mask = []
        segment_ids = []
        for idx, segments in enumerate(sup_ori_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        aug_input_ids = []
        aug_input_mask = []
        aug_segment_ids = []
        for idx, segments in enumerate(aug_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            aug_input_ids.append(_input_ids)
            aug_input_mask.append(_input_mask)
            aug_segment_ids.append(_segment_ids)

        return (input_ids, input_mask, segment_ids, aug_input_ids, aug_input_mask, aug_segment_ids, is_supervised)

    def _convert_y(self, y):
        label_set = set(y)
        if None in label_set:
            label_set -= {None}

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, "Number of unique `y`s exceeds `label_size`."
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

        # automatically set `label_to_id` for prediction
        if not self._label_to_id:
            self._label_to_id = {label: index for index, label in enumerate(self._id_to_label)}

        label_ids = []
        for label in y:
            if label is None:
                label_ids.append(-1)
                continue
            if label not in self._label_to_id:
                assert len(self._label_to_id) < self.label_size, "Number of unique labels exceeds `label_size`."
                self._label_to_id[label] = len(self._label_to_id)
                self._id_to_label.append(label)
            label_ids.append(self._label_to_id[label])
        return label_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }
        if kwargs.get("is_training"):
            self.placeholders.update({
                "aug_input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "aug_input_ids"),
                "aug_input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "aug_input_mask"),
                "aug_segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "aug_segment_ids"),
                "is_supervised": tf.placeholder(tf.float32, [None], "is_supervised"),
            })

    def _forward(self, is_training, placeholders, **kwargs):

        if not is_training:
            return super()._forward(is_training, placeholders, **kwargs)

        aug_input_ids = tf.boolean_mask(placeholders["aug_input_ids"], mask=(1.0 - placeholders["is_supervised"]), axis=0)
        aug_input_mask = tf.boolean_mask(placeholders["aug_input_mask"], mask=(1.0 - placeholders["is_supervised"]), axis=0)
        aug_segment_ids = tf.boolean_mask(placeholders["aug_segment_ids"], mask=(1.0 - placeholders["is_supervised"]), axis=0)
        input_ids = tf.concat([placeholders["input_ids"], aug_input_ids], axis=0)
        input_mask = tf.concat([placeholders["input_mask"], aug_input_mask], axis=0)
        segment_ids = tf.concat([placeholders["segment_ids"], aug_segment_ids], axis=0)
        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()

        label_ids = placeholders["label_ids"]
        is_expanded = tf.zeros_like(label_ids, dtype=tf.float32)
        batch_size = util.get_shape_list(aug_input_ids)[0]
        aug_is_expanded = tf.ones((batch_size), dtype=tf.float32)
        is_expanded = tf.concat([is_expanded, aug_is_expanded], axis=0)
        decoder = UDADecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            is_supervised=placeholders["is_supervised"],
            is_expanded=is_expanded,
            label_ids=label_ids,
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/seq_relationship",
            global_step=self._global_step,
            num_train_steps=self.total_steps,
            uda_softmax_temp=self._uda_softmax_temp,
            uda_confidence_thresh=self._uda_confidence_thresh,
            tsa_schedule=self._tsa_schedule,
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["supervised"], self.tensors["unsupervised"]]
        if from_tfrecords:
            ops.extend([self.placeholders["is_supervised"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_is_sup = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_is_sup = feed_dict[self.placeholders["is_supervised"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # accuracy
        batch_preds = output_arrays[0]
        accuracy = np.sum((batch_preds == batch_labels) * batch_is_sup) / np.sum(batch_is_sup)

        # supervised loss
        batch_sup_losses = output_arrays[1]
        sup_loss = np.mean(batch_sup_losses)

        # supervised loss
        batch_unsup_losses = output_arrays[2]
        unsup_loss = np.mean(batch_unsup_losses)

        info = ""
        info += ", accuracy %.4f" % accuracy
        info += ", supervised loss %.6f" % sup_loss
        info += ", unsupervised loss %.6f" % unsup_loss

        return info
