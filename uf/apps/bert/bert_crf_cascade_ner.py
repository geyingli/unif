import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_crf_ner import BERTCRFNER
from ..base.base_ner import NERModule
from ..crf.crf import CRFDecoder, viterbi_decode
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTCRFCascadeNER(BERTCRFNER, NERModule):
    """ Named entity recognization and classification on BERT with CRF. """
    _INFER_ATTRIBUTES = {
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "entity_types": "A list of strings that defines possible types of entities",
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
        entity_types=None,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(NERModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.entity_types = entity_types
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
        if is_parallel:
            assert self.entity_types, "Can't parse data on multi-processing when `entity_types` is None."

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
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
        if y:
            label_ids = self._convert_y(y, input_ids, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        if not self.entity_types:
            type_B_id = {}
        else:
            type_B_id = {
                entity_type: 1 + 4 * i
                for i, entity_type in enumerate(self.entity_types)
            }
        for idx, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if not isinstance(_y, dict):
                raise ValueError(
                    "Wrong input format of `y`. An untokenized example: "
                    "`y = [{\"Person\": [\"Trump\", \"Obama\"], "
                    "\"City\": [\"Washington D.C.\"], ...}, ...]`"
                )

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]

            # each type
            for _key in _y:

                # new type
                if _key not in type_B_id:
                    assert not self.entity_types, (
                        "Entity type `%s` not found in entity_types: %s."
                        % (_key, self.entity_types)
                    )
                    type_B_id[_key] = 1 + 4 * len(list(type_B_id.keys()))
                _entities = _y[_key]

                # each entity
                for _entity in _entities:
                    if isinstance(_entity, str):
                        _entity_tokens = self.tokenizer.tokenize(_entity)
                        _entity_ids = self.tokenizer.convert_tokens_to_ids(_entity_tokens)
                    elif isinstance(_entity, list):
                        if isinstance(_entity[0], str):
                            _entity_ids = self.tokenizer.convert_tokens_to_ids(_y)
                        else:
                            raise ValueError("Wrong input format (line %d): \"%s\". " % (idx, _entity))

                    # search and tag
                    start_positions = com.find_all_boyer_moore(_input_ids, _entity_ids)
                    if not start_positions:
                        tf.logging.warning(
                            "Failed to find the mapping of entity "
                            "to inputs at line %d. A possible "
                            "reason is that the entity span is "
                            "truncated due to the "
                            "`max_seq_length` setting."
                            % (idx)
                        )
                        continue

                    for start_position in start_positions:
                        end_position = start_position + len(_entity) - 1
                        if start_position == end_position:
                            _label_ids[start_position] = type_B_id[_key] + 3
                        else:
                            for i in range(start_position, end_position + 1):
                                _label_ids[i] = type_B_id[_key] + 1
                            _label_ids[start_position] = type_B_id[_key]
                            _label_ids[end_position] = type_B_id[_key] + 2

            label_ids.append(_label_ids)
        if not self.entity_types:
            self.entity_types = list(type_B_id.keys())
        return label_ids

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
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=placeholders["input_mask"],
            label_ids=placeholders["label_ids"],
            label_size=1 + len(self.entity_types) * 4,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, as_feature=False):
        ops = [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]
        if as_feature:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_logits = output_arrays[0]
        batch_transition_matrix = output_arrays[1]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ""
        for key in metrics:
            info += ", %s %.4f" % (key, metrics[key])
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"]]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # preds
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(_logits[:_input_length], transition_matrix)

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            _preds = {}
            for i, entity_type in enumerate(self.entity_types):
                _B_id = 1 + 4 * i
                _entities = self._get_entities(_viterbi_seq, _B_id)
                if _entities:
                    _preds[entity_type] = []

                    for _entity in _entities:
                        _start, _end = _entity[0], _entity[1]
                        if tokenized:
                            _entity_tokens = _tokens[_start: _end + 1]
                            _preds[entity_type].append(_entity_tokens)
                        else:
                            try:
                                _text_start = _mapping_start[_start]
                                _text_end = _mapping_end[_end]
                            except Exception:
                                continue
                            _entity_text = _text[_text_start: _text_end]
                            _preds[entity_type].append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs["preds"] = preds
        outputs["logits"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # f1
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data["input_mask"]
        labels = self.data["label_ids"]
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        metrics = self._get_cascade_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        for key in metrics:
            outputs[key] = metrics[key]
        outputs["loss"] = loss

        return outputs
