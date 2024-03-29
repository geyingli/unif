import numpy as np

from .transformer import Transformer, get_decay_power
from .._base_._base_mt import MTModule
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class TransformerMT(MTModule):
    """ Machine translation on Transformer. """

    def __init__(
        self,
        vocab_file,
        source_max_seq_length=64,
        target_max_seq_length=64,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(MTModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.source_max_seq_length = source_max_seq_length
        self.target_max_seq_length = target_max_seq_length
        self.truncate_method = truncate_method
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads

        self.tokenizer = WordPieceTokenizer(vocab_file, do_lower_case)
        self.decay_power = get_decay_power(num_hidden_layers)

        if "<s>" not in self.tokenizer.vocab:
            self.tokenizer.add("<s>")
            tf.logging.info("Add necessary token `<s>` into vocabulary.")
        if "</s>" not in self.tokenizer.vocab:
            self.tokenizer.add("</s>")
            tf.logging.info("Add necessary token `</s>` into vocabulary.")

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is not None, "`y` can't be None."

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            source_ids = self._convert_X(X_tokenized if tokenized else X, tokenized=tokenized)
            data["source_ids"] = np.array(source_ids, dtype=np.int32)
            n_inputs = len(source_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y is not None:
            target_ids = self._convert_y(y)
            data["target_ids"] = np.array(target_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized):
        source_ids = []

        for idx, sample in enumerate(X_target):
            try:
                _source_tokens = self._convert_x(sample, tokenized)
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))
            _source_ids = self.tokenizer.convert_tokens_to_ids(_source_tokens)

            com.truncate_segments([_source_ids], self.source_max_seq_length, truncate_method=self.truncate_method)

            if len(_source_ids) < self.source_max_seq_length:
                _source_ids.extend([0 for _ in range(self.source_max_seq_length - len(_source_ids))])
            source_ids.append(_source_ids)

        return source_ids

    def _convert_y(self, y):
        target_ids = []
        sos_id = self.tokenizer.convert_tokens_to_ids(["<s>"])[0]
        eos_id = self.tokenizer.convert_tokens_to_ids(["</s>"])[0]

        for _y in y:
            if isinstance(_y, str):
                _target_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_y))
            elif isinstance(_y, list):
                assert isinstance(_y[0], str), "Machine translation module only supports single sentence inputs."
                _target_ids = self.tokenizer.convert_tokens_to_ids(_y)

            com.truncate_segments([_target_ids], self.target_max_seq_length - 2, truncate_method=self.truncate_method)
            _target_ids = [sos_id] + _target_ids + [eos_id]

            if len(_target_ids) < self.target_max_seq_length:
                _target_ids.extend([0 for _ in range(self.target_max_seq_length - len(_target_ids))])
            target_ids.append(_target_ids)

        return target_ids

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "source_ids": tf.placeholder(tf.int32, [None, self.source_max_seq_length], "source_ids"),
            "target_ids": tf.placeholder(tf.int32, [None, self.target_max_seq_length], "target_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        model = Transformer(
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            source_ids=placeholders["source_ids"],
            target_ids=placeholders["target_ids"],
            sos_id=self.tokenizer.convert_tokens_to_ids(["<s>"])[0],
            sample_weight=placeholders.get("sample_weight"),
            hidden_size=self._hidden_size,
            num_blocks=self._num_hidden_layers,
            num_attention_heads=self._num_attention_heads,
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["target_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_target = output_arrays[-1]
        else:
            batch_target = feed_dict[self.placeholders["target_ids"]]

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
            for i in range(self.target_max_seq_length):
                if _pred_tokens[i] == "</s>":
                    _pred_tokens = _pred_tokens[:i]
                    break
            _pred_text = com.convert_tokens_to_text(_pred_tokens)
            preds.append(_pred_text)

        outputs = {}
        outputs["preds"] = preds

        return outputs

    def _get_score_ops(self):
        return [self.tensors["preds"], self.tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # accuracy
        preds = com.transform(output_arrays[0], n_inputs)
        target = self.data["target_ids"]
        labels = np.hstack((target[:, 1:], np.zeros((n_inputs, 1))))
        mask = (labels > 0)
        accuracy = np.sum((preds == labels) * mask) / np.sum(mask)

        # bleu
        bleu = self._get_bleu(preds, labels, mask)

        # rouge
        rouge = self._get_rouge(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["accuracy"] = accuracy
        outputs["bleu"] = bleu
        outputs["rouge"] = rouge
        outputs["loss"] = loss

        return outputs
