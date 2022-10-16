import collections
import numpy as np

from .bert import BERTEncoder, BERTConfig, get_decay_power
from .bert_mrc import BERTMRC
from ..base.base_mrc import MRCModule
from ..base.base import ClsDecoder, MRCDecoder
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class BERTVerifierMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on BERT, with a external front
    verifier. """
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=256,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        drop_pooler=False,
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._drop_pooler = drop_pooler
        self._on_predict = False

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
            label_ids, has_answer = self._convert_y(y, doc_ids, doc_text, doc_start, tokenized)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)
            data["has_answer"] = np.array(has_answer, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_y(self, y, doc_ids, doc_text, doc_start, tokenized=False):
        label_ids = []
        has_answer = []

        invalid_ids = []
        for idx, _y in enumerate(y):
            if _y is None:
                label_ids.append([0, 0])
                has_answer.append(0)
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

                start_positions = com.find_all_boyer_moore(_doc_ids, _answer_ids)
                if _overlap_time >= len(start_positions):
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(idx)
                    continue
                start_position = start_positions[_overlap_time]
                end_position = start_position + len(_answer_ids) - 1

            elif isinstance(_answer_text, list):
                start_position = _answer_start
                end_position = start_position + len(_answer_ids) - 1
                if _doc_ids[_answer_start: end_position + 1] != _answer_ids:
                    tf.logging.warning("Wrong `answer_start` at line %d. Ignored and set label to null text.")
                    label_ids.append([0, 0])
                    has_answer.append(0)
                    invalid_ids.append(idx)
                    continue

            label_ids.append([start_position + doc_start[idx], end_position + doc_start[idx]])
            has_answer.append(1)

        if invalid_ids:
            tf.logging.warning(
                "Failed to find the mapping of answer to inputs at "
                "line %s. A possible reason is that the answer spans "
                "are truncated due to the `max_seq_length` setting."
                % invalid_ids
            )

        return label_ids, has_answer

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "label_ids": tf.placeholder(tf.int32, [None, 2], "label_ids"),
            "has_answer": tf.placeholder(tf.int32, [None], "has_answer"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=split_placeholders["input_ids"],
            input_mask=split_placeholders["input_mask"],
            segment_ids=split_placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        verifier = ClsDecoder(
            is_training=is_training,
            input_tensor=encoder.get_pooled_output(),
            label_ids=split_placeholders["has_answer"],
            label_size=2,
            sample_weight=split_placeholders.get("sample_weight"),
            scope="cls/verifier",
            **kwargs,
        )
        if is_training:
            sample_weight = tf.cast(split_placeholders["has_answer"], tf.float32) * split_placeholders.get("sample_weight")
        else:
            sample_weight = split_placeholders.get("sample_weight")
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder.get_sequence_output(),
            label_ids=split_placeholders["label_ids"],
            sample_weight=sample_weight,
            scope="mrc",
            **kwargs,
        )

        verifier_total_loss, verifier_tensors = verifier.get_forward_outputs()
        decoder_total_loss, decoder_tensors = decoder.get_forward_outputs()

        total_loss = verifier_total_loss + decoder_total_loss
        tensors = collections.OrderedDict()
        for key in verifier_tensors:
            tensors["verifier_" + key] = verifier_tensors[key]
        for key in decoder_tensors:
            tensors["mrc_" + key] = decoder_tensors[key]

        return total_loss, tensors

    def _get_fit_ops(self, as_feature=False):
        ops = [
            self._tensors["verifier_preds"],
            self._tensors["verifier_losses"],
            self._tensors["mrc_preds"],
            self._tensors["mrc_losses"],
        ]
        if as_feature:
            ops.extend([self.placeholders["label_ids"]])
            ops.extend([self.placeholders["has_answer"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_labels = output_arrays[-2]
            batch_has_answer = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["label_ids"]]
            batch_has_answer = feed_dict[self.placeholders["has_answer"]]

        # verifier accuracy
        batch_has_answer_preds = output_arrays[0]
        has_answer_accuracy = np.mean(batch_has_answer_preds == batch_has_answer)

        # verifier loss
        batch_verifier_losses = output_arrays[1]
        verifier_loss = np.mean(batch_verifier_losses)

        # mrc exact match & f1
        batch_preds = output_arrays[2]
        for i in range(len(batch_has_answer_preds)):
            if batch_has_answer_preds[i] == 0:
                batch_preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(batch_preds, batch_labels)

        # mrc loss
        batch_losses = output_arrays[3]
        loss = np.mean(batch_losses)

        info = ""
        info += ", has_ans_accuracy %.4f" % has_answer_accuracy
        info += ", exact_match %.4f" % exact_match
        info += ", f1 %.4f" % f1
        info += ", verifier_loss %.6f" % verifier_loss
        info += ", mrc_loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [
            self._tensors["verifier_probs"],
            self._tensors["verifier_preds"],
            self._tensors["mrc_probs"],
            self._tensors["mrc_preds"],
        ]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier preds & probs
        verifier_probs = com.transform(output_arrays[0], n_inputs)[:, 1]
        verifier_preds = com.transform(output_arrays[1], n_inputs)

        # mrc preds & probs
        probs = com.transform(output_arrays[2], n_inputs)
        mrc_preds = com.transform(output_arrays[3], n_inputs)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
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
            self._tensors["verifier_preds"],
            self._tensors["verifier_losses"],
            self._tensors["mrc_preds"],
            self._tensors["mrc_losses"],
        ]

    def _get_score_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # verifier accuracy
        has_answer_preds = com.transform(output_arrays[0], n_inputs)
        has_answer_accuracy = np.mean(has_answer_preds == self.data["has_answer"])

        # verifier loss
        verifier_losses = com.transform(output_arrays[1], n_inputs)
        verifier_loss = np.mean(verifier_losses)

        # mrc exact match & f1
        preds = com.transform(output_arrays[2], n_inputs)
        for i in range(len(has_answer_preds)):
            if has_answer_preds[i] == 0:
                preds[i] = 0
        exact_match, f1 = self._get_em_and_f1(preds, self.data["label_ids"])

        # mrc loss
        losses = com.transform(output_arrays[3], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["has_ans_accuracy"] = has_answer_accuracy
        outputs["exact_match"] = exact_match
        outputs["f1"] = f1
        outputs["verifier_loss"] = verifier_loss
        outputs["mrc_loss"] = loss

        return outputs
