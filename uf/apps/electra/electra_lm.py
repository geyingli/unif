import numpy as np

from .electra import ELECTRA
from .._base_._base_lm import LMModule
from ..bert.bert_lm import BERTLM
from ..bert.bert import create_masked_lm_predictions
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class ELECTRALM(BERTLM, LMModule):
    """ Language modeling on ELECTRA. """

    def __init__(
        self,
        vocab_file,
        model_size="base",
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        generator_weight=1.0,
        discriminator_weight=50.0,
        max_predictions_per_seq=20,
        masked_lm_prob=0.15,
        do_whole_word_mask=False,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.generator_weight = generator_weight
        self.discriminator_weight = discriminator_weight
        self.masked_lm_prob = masked_lm_prob
        self.do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._model_size = model_size
        self._max_predictions_per_seq = max_predictions_per_seq

        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = "unsupported"

        if "[CLS]" not in self.tokenizer.vocab:
            self.tokenizer.add("[CLS]")
            tf.logging.info("Add necessary token `[CLS]` into vocabulary.")
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights) = self._convert_X(X_tokenized if tokenized else X, is_training, tokenized=tokenized)

            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            data["masked_lm_positions"] = np.array(masked_lm_positions, dtype=np.int32)
            data["masked_lm_ids"] = np.array(masked_lm_ids, dtype=np.int32)

            if is_training:
                data["masked_lm_weights"] = np.array(masked_lm_weights, dtype=np.float32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
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
                for _ in range(self._max_predictions_per_seq):
                    _masked_lm_ids.append(0)

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

        return (input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "masked_lm_positions": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_positions"),
            "masked_lm_ids": tf.placeholder(tf.int32, [None, self._max_predictions_per_seq], "masked_lm_ids"),
            "masked_lm_weights": tf.placeholder(tf.float32, [None, self._max_predictions_per_seq], "masked_lm_weights"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        model = ELECTRA(
            vocab_size=len(self.tokenizer.vocab),
            model_size=self._model_size,
            is_training=is_training,
            placeholders=placeholders,
            sample_weight=placeholders.get("sample_weight"),
            max_predictions_per_seq=self._max_predictions_per_seq,
            gen_weight=self.generator_weight,
            disc_weight=self.discriminator_weight,
            **kwargs,
        )

        return model.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [
            self._tensors["MLM_preds"],
            self._tensors["RTD_preds"],
            self._tensors["RTD_labels"],
            self._tensors["MLM_losses"],
            self._tensors["RTD_losses"],
        ]
        if from_tfrecords:
            ops.extend([
                self.placeholders["input_mask"],
                self.placeholders["masked_lm_positions"],
                self.placeholders["masked_lm_ids"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_input_mask = output_arrays[-3]
            batch_mlm_positions = output_arrays[-2]
            batch_mlm_labels = output_arrays[-1]
        else:
            batch_input_mask = feed_dict[self.placeholders["input_mask"]]
            batch_mlm_positions = feed_dict[self.placeholders["masked_lm_positions"]]
            batch_mlm_labels = feed_dict[self.placeholders["masked_lm_ids"]]

        # MLM accuracy
        batch_mlm_preds = output_arrays[0]
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) / batch_mlm_mask.sum()

        # RTD accuracy
        batch_rtd_preds = output_arrays[1]
        batch_rtd_labels = output_arrays[2]
        rtd_accuracy = np.sum((batch_rtd_preds == batch_rtd_labels) * batch_input_mask) / batch_input_mask.sum()

        # MLM loss
        batch_mlm_losses = output_arrays[3]
        mlm_loss = np.mean(batch_mlm_losses)

        # RTD loss
        batch_rtd_losses = output_arrays[4]
        rtd_loss = np.mean(batch_rtd_losses)

        info = ""
        info += ", MLM accuracy %.4f" % mlm_accuracy
        info += ", RTD accuracy %.4f" % rtd_accuracy
        info += ", MLM loss %.6f" % mlm_loss
        info += ", RTD loss %.6f" % rtd_loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["MLM_preds"], self._tensors["RTD_preds"], self._tensors["RTD_probs"]]

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

        # RTD preds
        rtd_preds = []
        input_ids = self.data["input_ids"]
        input_mask = self.data["input_mask"]
        all_preds = com.transform(output_arrays[1], n_inputs).tolist()
        for idx, _preds in enumerate(all_preds):
            _tokens = []
            _end = sum(input_mask[idx])
            for p_id, _id in enumerate(_preds[:_end]):
                if _id == 0:
                    _tokens.append(self.tokenizer.convert_ids_to_tokens([input_ids[idx][p_id]])[0])
                else:
                    _tokens.append("[REPLACED]")
            rtd_preds.append(_tokens)

        # RTD probs
        rtd_probs = com.transform(output_arrays[2], n_inputs)

        outputs = {}
        outputs["mlm_preds"] = mlm_preds
        outputs["rtd_preds"] = rtd_preds
        outputs["rtd_probs"] = rtd_probs

        return outputs
