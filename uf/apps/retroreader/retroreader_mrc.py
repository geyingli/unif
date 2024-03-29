import numpy as np

from .retroreader import RetroReaderDecoder, get_decay_power
from .._base_._base_mrc import MRCModule
from ..bert.bert_verifier_mrc import BERTVerifierMRC
from ..bert.bert import BERTEncoder, BERTConfig
from ..albert.albert import ALBERTEncoder, ALBERTConfig
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class RetroReaderMRC(BERTVerifierMRC, MRCModule):
    """ Machine reading comprehension on Retro-Reader. """

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=256,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        matching_mechanism="cross-attention",
        beta_1=0.5,
        beta_2=0.5,
        threshold=1.0,
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self._do_lower_case = do_lower_case
        self._matching_mechanism = matching_mechanism
        self._threshold = threshold

        self.bert_config = BERTConfig.from_json_file(config_file)
        assert matching_mechanism in ("cross-attention", "matching-attention"), (
            "Invalid value of `matching_machanism`: %s. Pick one from "
            "`cross-attention` and `matching-attention`."
        )
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
            (input_tokens, input_ids, input_mask, query_mask, segment_ids,
             doc_ids, doc_text, doc_start) = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["query_mask"] = np.array(query_mask, dtype=np.int32)
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
            label_ids, has_answer = self._convert_y(y, doc_ids, doc_text, doc_start, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)
            data["has_answer"] = np.array(has_answer, dtype=np.int32)

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
                raise ValueError(
                    "Wrong input format (%s): %s. "
                    "An untokenized example: "
                    "`X = [{\"doc\": \"...\", \"question\": \"...\", ...}, "
                    "...]`" % (sample, e)
                )

        input_tokens = []
        input_ids = []
        input_mask = []
        query_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _query_mask = [1]
            _segment_ids = [0]

            _doc_tokens = segments.pop("doc")
            segments = list(segments.values()) + [_doc_tokens]
            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)
            _doc_tokens = segments[-1]

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                if s_id == 0:
                    _query_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))
            _doc_start = len(_input_tokens) - len(_doc_tokens) - 1

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            _doc_ids = _input_ids[_doc_start: -1]

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)
            for _ in range(self.max_seq_length - len(_query_mask)):
                _query_mask.append(0)

            input_tokens.append(_input_tokens)
            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            query_mask.append(_query_mask)
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[idx]["doc"])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, query_mask, segment_ids, doc_ids, doc_text, doc_start)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "query_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "query_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, 2], "label_ids"),
            "has_answer": tf.placeholder(tf.int32, [None], "has_answer"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        sketchy_encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            **kwargs,
        )
        intensive_encoder = sketchy_encoder
        decoder = RetroReaderDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            sketchy_encoder=sketchy_encoder,
            intensive_encoder=intensive_encoder,
            query_mask=placeholders["query_mask"],
            label_ids=placeholders["label_ids"],
            has_answer=placeholders["has_answer"],
            sample_weight=placeholders.get("sample_weight"),
            matching_mechanism=self._matching_mechanism,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            threshold=self._threshold,
            trainable=True,
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [
            self.tensors["verifier_preds"],
            self.tensors["mrc_preds"],
            self.tensors["sketchy_losses"],
            self.tensors["intensive_losses"],
        ]
        if from_tfrecords:
            ops.extend([self.placeholders["label_ids"]])
            ops.extend([self.placeholders["has_answer"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_labels = output_arrays[-2]
            batch_has_answer = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_ids"]]
            batch_has_answer = feed_dict[self.placeholders["has_answer"]]

        # verifier accuracy
        batch_has_answer_preds = output_arrays[0]
        has_answer_accuracy = np.mean(batch_has_answer_preds == batch_has_answer)

        # mrc exact match & f1
        batch_preds = output_arrays[1]
        for i in range(len(batch_has_answer_preds)):
            if batch_has_answer_preds[i] == 0:
                batch_preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # sketchy loss
        batch_sketchy_losses = output_arrays[2]
        sketchy_loss = np.mean(batch_sketchy_losses)

        # intensive loss
        batch_intensive_losses = output_arrays[3]
        intensive_loss = np.mean(batch_intensive_losses)

        info = ""
        info += ", has_ans_accuracy %.4f" % has_answer_accuracy
        info += ", exact_match %.4f" % exact_match
        info += ", f1 %.4f" % f1
        info += ", sketchy_loss %.6f" % sketchy_loss
        info += ", intensive_loss %.6f" % intensive_loss

        return info

    def _get_predict_ops(self):
        return [
            self.tensors["verifier_probs"],
            self.tensors["verifier_preds"],
            self.tensors["mrc_probs"],
            self.tensors["mrc_preds"],
        ]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # verifier preds & probs
        verifier_probs = com.transform(output_arrays[0], n_inputs)
        verifier_preds = com.transform(output_arrays[1], n_inputs)

        # mrc preds & probs
        preds = []
        probs = com.transform(output_arrays[2], n_inputs)
        mrc_preds = com.transform(output_arrays[3], n_inputs)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        for idx, _preds in enumerate(mrc_preds):
            _start, _end = int(_preds[0]), int(_preds[1])
            if verifier_preds[idx] == 0 or _start == 0 or _end == 0 or _start > _end:
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
        outputs["verifier_probs"] = verifier_probs
        outputs["verifier_preds"] = verifier_preds
        outputs["mrc_probs"] = probs
        outputs["mrc_preds"] = preds

        return outputs

    def _get_score_ops(self):
        return [
            self.tensors["verifier_preds"],
            self.tensors["mrc_preds"],
            self.tensors["sketchy_losses"],
            self.tensors["intensive_losses"],
        ]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # verifier accuracy
        has_answer_preds = com.transform(output_arrays[0], n_inputs)
        has_answer_accuracy = np.mean(has_answer_preds == self.data["has_answer"])

        # mrc exact match & f1
        preds = com.transform(output_arrays[1], n_inputs)
        for i in range(len(has_answer_preds)):
            if has_answer_preds[i] == 0:
                preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(preds, self.data["label_ids"])

        # sketchy loss
        sketchy_losses = com.transform(output_arrays[2], n_inputs)
        sketchy_loss = np.mean(sketchy_losses)

        # intensive loss
        intensive_losses = com.transform(output_arrays[3], n_inputs)
        intensive_loss = np.mean(intensive_losses)

        outputs = {}
        outputs["has_ans_accuracy"] = has_answer_accuracy
        outputs["exact_match"] = exact_match
        outputs["f1"] = f1
        outputs["sketchy_loss"] = sketchy_loss
        outputs["intensive_loss"] = intensive_loss

        return outputs
