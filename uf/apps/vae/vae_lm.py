import numpy as np

from .vae import VAE, get_decay_power
from .._base_._base_lm import LMModule
from ..bert.bert_classifier import BERTClassifier
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class VAELM(BERTClassifier, LMModule):
    """ Text generator in VAE structure. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        reduced_size=64,
        topic_size=1024,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._reduced_size = reduced_size
        self._topic_size = topic_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._bias = 0

        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(num_hidden_layers)

        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add("[SEP]")
            tf.logging.info("Add necessary token `[SEP]` into vocabulary.")

    def predict(self, X=None, X_tokenized=None, batch_size=8, bias=0):
        """ Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            bias: float. The absolute value of the upper and lower range
              of random uniform noise for text generation.
        Returns:
            A dict object of model outputs.
        """

        if bias != self._bias:
            self._bias = bias
            self._session_mode = None

        return super(LMModule, self).predict(X, X_tokenized, batch_size)

    def export(self, export_dir, bias=0, rename_inputs=None, rename_outputs=None, ignore_outputs=None):
        """ Export model into SavedModel files.

        Args:
            export_dir: str. Directory to which the model is saved.
            bias: float. The absolute value of the upper and lower range
              of random uniform noise for text generation.
            rename_inputs: dict. Mapping of original name to target name.
            rename_outputs: dict. Mapping of original name to target name.
            ignore_outputs: list. Name of outputs to ignore.
        Returns:
            None
        """

        if bias != self._bias:
            self._bias = bias
            self._session_mode = None

        return super(LMModule, self).export(export_dir, rename_inputs, rename_outputs, ignore_outputs)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            input_ids, input_mask, segment_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["segment_ids"] = np.array(segment_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

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
            except Exception:
                raise ValueError("Wrong input format (line %d): \"%s\". " % (idx, sample))

        input_ids = []
        input_mask = []
        segment_ids = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = []
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            com.truncate_segments(segments, self.max_seq_length - len(segments), truncate_method=self.truncate_method)
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

        return input_ids, input_mask, segment_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "input_mask": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_mask"),
            "segment_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "segment_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        model = VAE(
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            sample_weight=placeholders.get("sample_weight"),
            reduced_size=self._reduced_size,
            topic_size=self._topic_size,
            hidden_size=self._hidden_size,
            num_hidden_layers=self._num_hidden_layers,
            num_attention_heads=self._num_attention_heads,
            bias=self._bias,
            **kwargs,
        )
        return model.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self._tensors["preds"], self._tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["input_ids"], self.placeholders["input_mask"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_labels = output_arrays[-2]
            batch_mask = output_arrays[-1]
        else:
            batch_labels = feed_dict[self.placeholders["input_ids"]]
            batch_mask = feed_dict[self.placeholders["input_mask"]]

        # accuracy
        batch_preds = output_arrays[0]
        accuracy = np.sum((batch_preds == batch_labels) * batch_mask) / batch_mask.sum()

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", accuracy %.4f" % accuracy
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["miu"], self._tensors["sigma"], self._tensors["preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # miu
        miu = com.transform(output_arrays[0], n_inputs)

        # sigma
        sigma = com.transform(output_arrays[1], n_inputs)

        # preds
        all_preds = com.transform(output_arrays[2], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            for i in range(self.max_seq_length):
                if _pred_ids[i] == 0:
                    _pred_tokens = _pred_tokens[:i]
                    break
            preds.append(_pred_tokens)

        outputs = {}
        outputs["miu"] = miu
        outputs["sigma"] = sigma
        outputs["preds"] = preds

        return outputs
