import collections
import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_classifier import BERTClassifier
from .._base_._base_ner import NERModule
from .._base_._base_ import SeqClsDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTNER(BERTClassifier, NERModule):
    """ Named entity recognition on BERT. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(NERModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case

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

        if is_training:
            assert y is not None, "`y` can't be None."

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            X_target = X_tokenized if tokenized else X
            input_tokens, input_ids, input_mask, segment_ids = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y is not None:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        for idx, segments in enumerate(segment_input_tokens):
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

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_tokens, input_ids, input_mask, segment_ids

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        for idx, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if isinstance(_y, str):
                _entity_tokens = self.tokenizer.tokenize(_y)
                _entity_ids = [self.tokenizer.convert_tokens_to_ids(_entity_tokens)]
            elif isinstance(_y, list):
                if isinstance(_y[0], str):
                    if tokenized:
                        _entity_ids = [self.tokenizer.convert_tokens_to_ids(_y)]
                    else:
                        _entity_ids = []
                        for _entity in _y:
                            _entity_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_entity)))
                elif isinstance(_y[0], list):
                    _entity_ids = []
                    for _entity in _y:
                        _entity_ids.append(self.tokenizer.convert_tokens_to_ids(_entity))
            else:
                raise ValueError("`y` should be a list of entity strings.")

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]
            for _entity in _entity_ids:
                start_positions = com.find_all_boyer_moore(_input_ids, _entity)
                if not start_positions:
                    tf.logging.warning(
                        "Failed to find the mapping of entity to "
                        "inputs at line %d. A possible reason is "
                        "that the entity span is truncated due "
                        "to the `max_seq_length` setting." % (idx)
                    )
                    continue

                for start_position in start_positions:
                    end_position = start_position + len(_entity) - 1
                    if start_position == end_position:
                        _label_ids[start_position] = self.S_ID
                    else:
                        for i in range(start_position, end_position + 1):
                            _label_ids[i] = self.I_ID
                        _label_ids[start_position] = self.B_ID
                        _label_ids[end_position] = self.E_ID

            label_ids.append(_label_ids)

        return label_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
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
            label_size=5,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_preds = output_arrays[0]
        f1_token, f1_entity = self._get_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", f1/token %.4f" % f1_token
        info += ", f1/entity %.4f" % f1_entity
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        all_preds = np.argmax(probs, axis=-1)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(all_preds)):
            _preds = all_preds[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _entities = self._get_entities(_preds[:_input_length])
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except Exception:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["preds"], self._tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # f1
        preds = com.transform(output_arrays[0], n_inputs)
        labels = self.data["label_ids"]
        mask = self.data["input_mask"]
        f1_token, f1_entity = self._get_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["f1/token"] = f1_token
        outputs["f1/entity"] = f1_entity
        outputs["loss"] = loss

        return outputs
