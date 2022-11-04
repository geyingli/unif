import numpy as np

from .gpt2 import GPT2, GPT2Config, get_decay_power
from .._base_._base_lm import LMModule
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class GPT2LM(LMModule):
    """ Language modeling on GPT-2. """

    def __init__(
        self,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=1024,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._given = 1

        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.gpt2_config = GPT2Config(
            n_vocab=len(self.tokenizer.vocab),
            n_predict=max_seq_length,
            n_ctx=max_position_embeddings,
            n_embed=hidden_size,
            n_head=num_attention_heads,
            n_layer=num_hidden_layers,
        )
        self.decay_power = get_decay_power(num_hidden_layers)

        if "<eos>" not in self.tokenizer.vocab:
            self.tokenizer.add("<eos>")
            self.gpt2_config.n_vocab += 1
            tf.logging.info("Add necessary token `<eos>` into vocabulary.")
        self._eos_id = self.tokenizer.convert_tokens_to_ids(["<eos>"])[0]

    def predict(self, X=None, X_tokenized=None, batch_size=8, given=1):
        """ Inference on the model.

        Args:
            X: list. A list object consisting untokenized inputs.
            X_tokenized: list. A list object consisting tokenized inputs.
              Either `X` or `X_tokenized` should be None.
            batch_size: int. The size of batch in each step.
            given: int. The number of already known tokens.
        Returns:
            A dict object of model outputs.
        """

        if given != self._given:
            self._given = given
            self._graph_mode = None

        return super(LMModule, self).predict(X, X_tokenized, batch_size)

    def export(self, export_dir, given=1, rename_inputs=None, rename_outputs=None, ignore_outputs=None):
        """ Export model into SavedModel files.

        Args:
            export_dir: str. Directory to which the model is saved.
            given: int. The number of already known tokens.
            rename_inputs: dict. Mapping of original name to target name.
            rename_outputs: dict. Mapping of original name to target name.
            ignore_outputs: list. Name of outputs to ignore.
        Returns:
            None
        """

        if given != self._given:
            self._given = given
            self._graph_mode = None

        return super(LMModule, self).export(export_dir, rename_inputs, rename_outputs, ignore_outputs)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            input_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        input_ids = []

        for idx, sample in enumerate(X_target):
            try:
                _input_tokens = self._convert_x(sample, tokenized)
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))
            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            com.truncate_segments([_input_ids], self.max_seq_length - 1, truncate_method=self.truncate_method)
            _input_ids.append(self._eos_id)

            if len(_input_ids) < self.max_seq_length:
                _input_ids.extend([0 for _ in range(self.max_seq_length - len(_input_ids))])
            input_ids.append(_input_ids)

        return input_ids

    def _convert_x(self, x, tokenized):
        if not tokenized:
            # deal with general inputs
            if isinstance(x, str):
                return self.tokenizer.tokenize(x)

        # deal with tokenized inputs
        elif isinstance(x[0], str):
            return x

        # deal with tokenized and multiple inputs
        raise ValueError("GPT2 only supports single sentence inputs.")

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        model = GPT2(
            hparams=self.gpt2_config,
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            sample_weight=placeholders.get("sample_weight"),
            given=self._given,
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["input_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_target = output_arrays[-1]
        else:
            batch_target = feed_dict[self.placeholders["input_ids"]]

        # accuracy
        batch_preds = output_arrays[0]
        batch_labels = np.hstack((batch_target[:, 1:], np.zeros((self.batch_size, 1))))
        batch_mask = (batch_labels > 0)
        accuracy = np.sum((batch_preds == batch_labels) * batch_mask) / np.sum(batch_mask)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", accuracy %.4f" % accuracy
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # preds
        all_preds = com.transform(output_arrays[0], n_inputs).tolist()
        preds = []
        for _pred_ids in all_preds:
            _pred_tokens = self.tokenizer.convert_ids_to_tokens(_pred_ids)
            for i in range(self.max_seq_length):
                if _pred_tokens[i] == "<eos>":
                    _pred_tokens = _pred_tokens[:i]
                    break
            _pred_text = com.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs["preds"] = preds

        return outputs
