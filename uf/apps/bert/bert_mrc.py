import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_classifier import BERTClassifier
from ..base.base_mrc import MRCModule
from ..base.base import MRCDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTMRC(BERTClassifier, MRCModule):
    """ Machine reading comprehension on BERT. """
    
    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

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
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, segment_ids, doc_ids, doc_text, doc_start) = self._convert_X(X_target, tokenized=tokenized)
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
            label_ids = self._convert_y(y, doc_ids, doc_text, doc_start, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception:
                raise ValueError(
                    "Wrong input format (line %d): \"%s\". "
                    "An untokenized example: "
                    "`X = [{\"doc\": \"...\", \"question\": \"...\", ...}, "
                    "...]`" % (idx, sample)
                )

        input_tokens = []
        input_ids = []
        input_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]

            _doc_tokens = segments.pop("doc")
            segments = list(segments.values()) + [_doc_tokens]
            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            _doc_tokens = segments[-1]

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))
            _doc_start = len(_input_tokens) - len(_doc_tokens) - 1

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _doc_ids = _input_ids[_doc_start: -1]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[idx]["doc"])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, segment_ids, doc_ids, doc_text, doc_start)

    def _convert_x(self, x, tokenized):
        output = {}

        assert isinstance(x, dict) and "doc" in x

        for key in x:
            if not tokenized:
                # deal with general inputs
                output[key] = self.tokenizer.tokenize(x[key])
                continue

            # deal with tokenized inputs
            output[key] = x[key]

        return output

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []

        invalid_ids = []
        for idx, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                continue

            if not isinstance(_y, dict) or "text" not in _y or "answer_start" not in _y:
                raise ValueError(
                    "Wrong input format of `y`. An untokenized example: "
                    "`y = [{\"text\": \"Obama\", \"answer_start\": 12}, "
                    "None, ...]`"
                )

            _answer_text = _y["text"]
            _answer_start = _y["answer_start"]
            _doc_ids = doc_ids[idx]

            # tokenization
            if isinstance(_answer_text, str):
                _answer_tokens = self.tokenizer.tokenize(_answer_text)
                _answer_ids = self.tokenizer.convert_tokens_to_ids(_answer_tokens)
            elif isinstance(_answer_text, list):
                assert tokenized, "%s does not support multiple answer spans." % self.__class__.__name__
                _answer_tokens = _answer_text
                _answer_ids = self.tokenizer.convert_tokens_to_ids(_answer_text)
            else:
                raise ValueError("Invalid answer text at line %d: `%s`." % (idx, _answer_text))

            if isinstance(_answer_text, str):
                _overlap_time = len(com.find_all_boyer_moore(doc_text[idx][:_answer_start], _answer_text))
                try:
                    start_position = com.find_all_boyer_moore(_doc_ids, _answer_ids)[_overlap_time]
                    end_position = start_position + len(_answer_ids) - 1
                except IndexError:
                    label_ids.append([0, 0])
                    invalid_ids.append(idx)
                    continue
            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning("Wrong `answer_start` at line %d. Ignored and set label to null text.")
                    label_ids.append([0, 0])
                    invalid_ids.append(idx)
                    continue

            label_ids.append([start_position + doc_start[idx], end_position + doc_start[idx]])

        if invalid_ids:
            tf.logging.warning(
                "Failed to find the mapping of answer to inputs at "
                "line %s. A possible reason is that the answer spans "
                "are truncated due to the `max_seq_length` setting."
                % invalid_ids
            )

        return label_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, 2], "label_ids"),
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
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=placeholders["label_ids"],
            sample_weight=placeholders.get("sample_weight"),
            scope="mrc",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_labels = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # exact match & f1
        batch_preds = output_arrays[0]
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", exact_match %.4f" % exact_match
        info += ", f1 %.4f" % f1
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["probs"], self._tensors["preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        batch_preds = com.transform(output_arrays[1], n_inputs)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for idx, _preds in enumerate(batch_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if _start == 0 or _end == 0 or _start > _end:
                preds.append(None)
                continue
            _tokens = tokens[idx]

            if tokenized:
                _span_tokens = _tokens[_start: _end + 1]
                preds.append(_span_tokens)
            else:
                _sample = text[idx]
                _text = [_sample[key] for key in _sample if key != "doc"]
                _text.append(_sample["doc"])
                _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

                try:
                    _text_start = _mapping_start[_start]
                    _text_end = _mapping_end[_end]
                except Exception:
                    preds.append(None)
                    continue
                _span_text = _text[_text_start: _text_end]
                preds.append(_span_text)

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["preds"], self._tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # exact match & f1
        batch_preds = com.transform(output_arrays[0], n_inputs)
        batch_labels = self.data["label_ids"]
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["exact_match"] = exact_match
        outputs["f1"] = f1
        outputs["loss"] = loss

        return outputs
