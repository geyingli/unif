import numpy as np

from .._base_._base_lm import LMModule
from ..bert.bert_lm import BERTLM
from ..bert.bert import BERTDecoder, BERTConfig, create_instances_from_document
from .unilm import UniLMEncoder, create_masked_lm_predictions, get_decay_power
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class UniLM(LMModule):
    """ Language modeling on UniLM. """

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
        do_sample_next_sentence=True,
        max_predictions_per_seq=20,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        do_whole_word_mask=False,
        mode="bi",
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.do_sample_next_sentence = do_sample_next_sentence
        self.masked_lm_prob = masked_lm_prob
        self.short_seq_prob = short_seq_prob
        self.do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._drop_pooler = drop_pooler
        self._max_predictions_per_seq = max_predictions_per_seq
        self.mode = mode

        assert mode in ("bi", "l2r", "r2l", "s2s"), (
            "Wrong value of `mode`: %s. Pick one from `bi` (bidirectional), "
            "`l2r` (left-to-right), `r2l` (right-to-left) and `s2s` (seq-to-seq)." % mode
        )
        tf.logging.info("LM Mode: `%s`. Use method `.to_mode()` to convert it into `bi`, `l2r`, `r2l` or `s2s`." % mode)
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

    def to_mode(self, mode):
        """ Switch the mode of UniLM.

        Args:
            mode: string. One of `bi` (bidirectional), `l2r` (left-to-right),
              `r2l` (right-to-left) and `s2s` (seq-to-seq).
        Returns:
            None
        """
        assert mode in ("bi", "l2r", "r2l", "s2s"), (
            "Wrong value of `mode`: %s. Pick one from `bi` (bidirectional), "
            "`l2r` (left-to-right), `r2l` (right-to-left) and "
            "`s2s` (seq-to-seq)." % mode
        )

        if mode != self.mode:
            self._graph_mode = None
        self.mode = mode

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            if self.mode == "bi":
                if y is not None:
                    assert not self.do_sample_next_sentence, "`y` should be None when `do_sample_next_sentence` is True."
                else:
                    assert self.do_sample_next_sentence, "`y` can't be None when `do_sample_next_sentence` is False."
            else:
                assert y is None, "Only training of bidirectional LM is supervised. `y` should be None."

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights,
             next_sentence_labels) = self._convert_X(X_tokenized if tokenized else X, is_training, tokenized=tokenized)

            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["masked_lm_positions"] = np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data["masked_lm_ids"] = np.array(masked_lm_ids, dtype=np.int32)
                data["masked_lm_weights"] = np.array(masked_lm_weights, dtype=np.float32)
                data["next_sentence_labels"] = np.array(next_sentence_labels, dtype=np.int32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y is not None:
            next_sentence_labels = self._convert_y(y)
            data["next_sentence_labels"] = np.array(next_sentence_labels, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            if self.mode in ("l2r", "r2l"):
                info = "`l2r` or `r2l` only supports single sentence inputs."
                if not tokenized:
                    assert isinstance(sample, str), info
                else:
                    assert isinstance(sample[0], str), info
            elif self.mode == "s2s":
                info = "`s2s` only supports 2-sentence inputs."
                assert len(sample) == 2, info
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))

        input_ids = []
        input_mask = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        next_sentence_labels = []

        # random sampling of next sentence
        if is_training and self.mode == "bi" and self.do_sample_next_sentence:
            new_segment_input_tokens = []
            for idx in range(len(segment_input_tokens)):
                instances = create_instances_from_document(
                    all_documents=segment_input_tokens,
                    document_index=idx,
                    max_seq_length=self.max_seq_length - 3,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    short_seq_prob=self.short_seq_prob,
                    vocab_words=list(self.tokenizer.vocab.keys()),
                )
                for (segments, is_random_next) in instances:
                    new_segment_input_tokens.append(segments)
                    next_sentence_labels.append(is_random_next)
            segment_input_tokens = new_segment_input_tokens
        else:
            next_sentence_labels = [1] * len(segment_input_tokens)

        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _masked_lm_positions = []
            _masked_lm_ids = []
            _masked_lm_weights = []

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            # special values for `input_mask`
            if self.mode == "s2s":
                _input_mask = [len(segments[0]) + 2] * (len(segments[0]) + 2)
                for i in range(len(segments[1]) + 1):
                    _input_mask.append(_input_mask[0] + i + 1)

            # random sampling of masked tokens
            if is_training:
                if (idx + 1) % 10000 == 0:
                    tf.logging.info("Sampling masks of input %d" % (idx + 1))
                _input_tokens, _masked_lm_positions, _masked_lm_labels = create_masked_lm_predictions(
                    tokens=_input_tokens,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    vocab_words=list(self.tokenizer.vocab.keys()),
                    do_whole_word_mask=self.do_whole_word_mask,
                )
                _masked_lm_ids = self.tokenizer.convert_tokens_to_ids(_masked_lm_labels)
                _masked_lm_weights = [1.0] * len(_masked_lm_positions)

                # padding
                for _ in range(self._max_predictions_per_seq - len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)
                    _masked_lm_ids.append(0)
                    _masked_lm_weights.append(0.0)
            else:
                # `masked_lm_positions` is required for both training
                # and inference of BERT language modeling.
                for i in range(len(_input_tokens)):
                    if _input_tokens[i] == "[MASK]":
                        _masked_lm_positions.append(i)

                # padding
                for _ in range(self._max_predictions_per_seq - len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, next_sentence_labels)

    def _convert_y(self, y):
        label_set = set(y)

        # automatically set `label_size`
        if self.label_size:
            assert len(label_set) <= self.label_size, "Number of unique `y`s exceeds 2."
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
            "masked_lm_positions": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_positions"),
            "masked_lm_ids": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_ids"),
            "masked_lm_weights": tf.placeholder(tf.float32, [None, self._max_predictions_per_seq], "masked_lm_weights"),
            "next_sentence_labels": tf.placeholder(tf.int32, [None], "next_sentence_labels"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = UniLMEncoder(
            mode=self.mode,
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            drop_pooler=self._drop_pooler,
            **kwargs,
        )
        decoder = BERTDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            encoder=encoder,
            masked_lm_positions=placeholders["masked_lm_positions"],
            masked_lm_ids=placeholders["masked_lm_ids"],
            masked_lm_weights=placeholders["masked_lm_weights"],
            next_sentence_labels=placeholders["next_sentence_labels"],
            sample_weight=placeholders.get("sample_weight"),
            scope_lm="cls/predictions",
            scope_cls="cls/seq_relationship",
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["MLM_preds"], self.tensors["MLM_losses"]]
        if self.mode == "bi":
            ops.extend([self.tensors["NSP_preds"], self.tensors["NSP_losses"]])
        if from_tfrecords:
            ops.extend([self.placeholders["masked_lm_positions"], self.placeholders["masked_lm_ids"], self.placeholders["next_sentence_labels"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mlm_positions = output_arrays[-3]
            batch_mlm_labels = output_arrays[-2]
            batch_nsp_labels = output_arrays[-1]
        else:
            batch_mlm_positions = feed_dict[self.placeholders["masked_lm_positions"]]
            batch_mlm_labels = feed_dict[self.placeholders["masked_lm_ids"]]
            batch_nsp_labels = feed_dict[self.placeholders["next_sentence_labels"]]

        # MLM accuracy
        batch_mlm_preds = output_arrays[0]
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) / batch_mlm_mask.sum()

        # MLM loss
        batch_mlm_losses = output_arrays[1]
        mlm_loss = np.mean(batch_mlm_losses)

        info = ""
        info += ", MLM accuracy %.4f" % mlm_accuracy
        info += ", MLM loss %.6f" % mlm_loss

        if self.mode == "bi":

            # NSP accuracy
            batch_nsp_preds = output_arrays[2]
            nsp_accuracy = np.mean(batch_nsp_preds == batch_nsp_labels)

            # NSP loss
            batch_nsp_losses = output_arrays[3]
            nsp_loss = np.mean(batch_nsp_losses)

            info += ", NSP accuracy %.4f" % nsp_accuracy
            info += ", NSP loss %.6f" % nsp_loss

        return info

    def _get_predict_ops(self):
        ops = [self.tensors["MLM_preds"]]
        if self.mode == "bi":
            ops += [self.tensors["NSP_preds"], self.tensors["NSP_probs"]]
        return ops

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # MLM preds
        mlm_preds = []
        mlm_positions = self.data["masked_lm_positions"]
        all_preds = com.transform(output_arrays[0], n_inputs)
        for idx, _preds in enumerate(all_preds):
            _ids = []
            for p_id, _id in enumerate(_preds):
                if mlm_positions[idx][p_id] == 0:
                    break
                _ids.append(_id)
            mlm_preds.append(self.tokenizer.convert_ids_to_tokens(_ids))

        outputs = {}
        outputs["mlm_preds"] = mlm_preds

        if self.mode == "bi":

            # NSP preds
            nsp_preds = com.transform(output_arrays[1], n_inputs).tolist()

            # NSP probs
            nsp_probs = com.transform(output_arrays[2], n_inputs)

            outputs["nsp_preds"] = nsp_preds
            outputs["nsp_probs"] = nsp_probs

        return outputs
