import numpy as np

from .._base_._base_lm import LMModule
from ..bert.bert_lm import BERTLM
from ..bert.bert import BERTDecoder, BERTConfig, create_instances_from_document
from .unilm import UniLMEncoder, create_masked_lm_predictions, get_decay_power
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class UniLMPrompt(BERTLM, LMModule):
    """ Prompt Language modeling on UniLM. """

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        drop_pooler=False,
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

        assert X is None, "`%s` only supports tokenized inputs. Please pass `X_tokenized`." % (self.__class__.__name__)
        if is_training:
            if not self.masked_lm_prob:
                assert y is not None, "`y` can't be None when `masked_lm_prob` is 0." % (self.__class__.__name__)

        n_inputs = None
        data = {}

        # convert X
        if X_tokenized is not None:
            (input_ids, input_mask, segment_ids, prompt_length, 
            masked_lm_positions, masked_lm_ids, masked_lm_weights) = self._convert_X(X_tokenized, y, is_training)

            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["prompt_length"] = np.array(prompt_length, dtype=np.int32)
            data["masked_lm_positions"] = np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data["masked_lm_ids"] = np.array(masked_lm_ids, dtype=np.int32)
                data["masked_lm_weights"] = np.array(masked_lm_weights, dtype=np.float32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, y, is_training):
        input_ids = []
        input_mask = []
        segment_ids = []
        prompt_length = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []

        for idx, _X in enumerate(X_target):
            _prompts = _X[:-1]
            _text = _X[-1:]
            if self.mode == "s2s":
                _prompts = _X[:-2]
                _text = _X[-2:]
            _y = y[idx]

            # process prompt
            _prompt_tokens = ["[CLS]"]
            for i, segment in enumerate(_prompts):
                _prompt_tokens += segment
                _prompt_tokens.append("[SEP]")
            _prompt_length = len(_prompt_tokens)

            # process text
            com.truncate_segments(_text, self.max_seq_length - _prompt_length - len(_text), truncate_method=self.truncate_method)
            _text_tokens = []
            for i, segment in enumerate(_text):
                _text_tokens += segment
                _text_tokens.append("[SEP]")

            # process label
            _masked_lm_positions = []
            _masked_lm_labels = []
            if is_training and self.masked_lm_prob > 0:
                if (idx + 1) % 10000 == 0:
                    tf.logging.info("Sampling masks of input %d" % (idx + 1))
                _text_tokens, _masked_lm_positions, _masked_lm_labels = create_masked_lm_predictions(
                    tokens=_text_tokens,
                    masked_lm_prob=self.masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    vocab_words=list(self.tokenizer.vocab.keys()),
                    do_whole_word_mask=self.do_whole_word_mask,
                )
                _masked_lm_positions = [p + _prompt_length for p in _masked_lm_positions]
            else:
                j = 0
                for i in range(len(_text_tokens)):
                    if _text_tokens[i] == "[MASK]":
                        _masked_lm_positions.append(i + _prompt_length)
                        _masked_lm_labels.append(_y[j])
                        j += 1

            _input_ids = self.tokenizer.convert_tokens_to_ids(_prompt_tokens + _text_tokens)
            _input_mask = [1] * len(_input_ids)
            _segment_ids = [0] * len(_input_ids)
            _masked_lm_ids = self.tokenizer.convert_tokens_to_ids(_masked_lm_labels)
            _masked_lm_weights = [1.0] * len(_masked_lm_positions)

            # special values for `input_mask`
            if self.mode == "r2l":
                _input_mask = [_prompt_length] * (len(_input_ids) - len(_text[-1]) - 1)
                for i in range(len(_text[-1]) + 1):
                    _input_mask.append(_prompt_length + i)
            elif self.mode == "s2s":
                _input_mask = [len(_input_ids) - len(_X[-1]) - 1] * (len(_input_ids) - len(_text[-1]) - 1)
                for i in range(len(_text[-1]) + 1):
                    _input_mask.append(_input_mask[0] + i + 1)

            # padding
            for _ in range(self._max_predictions_per_seq - len(_masked_lm_positions)):
                _masked_lm_positions.append(0)
                _masked_lm_ids.append(0)
                _masked_lm_weights.append(0.0)
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            prompt_length.append(_prompt_length)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids, input_mask, segment_ids, prompt_length, masked_lm_positions, masked_lm_ids, masked_lm_weights)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "prompt_length": tf.placeholder(tf.int32, [None], "prompt_length"),
            "masked_lm_positions": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_positions"),
            "masked_lm_ids": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_ids"),
            "masked_lm_weights": tf.placeholder(tf.float32, [None, self._max_predictions_per_seq], "masked_lm_weights"),
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
            prompt_length=placeholders["prompt_length"],
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
            sample_weight=placeholders.get("sample_weight"),
            scope_lm="cls/predictions",
            scope_cls="cls/seq_relationship",
            **kwargs,
        )
        train_loss, tensors = decoder.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["MLM_preds"], self.tensors["MLM_losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["masked_lm_positions"], self.placeholders["masked_lm_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mlm_positions = output_arrays[-2]
            batch_mlm_labels = output_arrays[-1]
        else:
            batch_mlm_positions = feed_dict[self.placeholders["masked_lm_positions"]]
            batch_mlm_labels = feed_dict[self.placeholders["masked_lm_ids"]]

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

        return info

    def _get_predict_ops(self):
        ops = [self.tensors["MLM_preds"]]
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

        return outputs
