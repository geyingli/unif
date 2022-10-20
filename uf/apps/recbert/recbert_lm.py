import numpy as np

from .recbert import RecBERT, sample_wrong_tokens
from .._base_._base_lm import LMModule
from ..bert.bert import BERTConfig, get_decay_power
from ...token import WordPieceTokenizer
from ...third import tf
from ... import com


class RecBERTLM(LMModule):
    """ Language modeling on RecBERT. """

    def __init__(
        self,
        config_file,
        vocab_file,
        max_seq_length=128,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        add_prob=0.1,
        del_prob=0.1,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._do_lower_case = do_lower_case
        self._add_prob = add_prob
        self._del_prob = del_prob

        assert add_prob <= 0.5, "The value of `add_prob` should be larger than 0 and smaller than 1/2."
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

        assert y is None, ("%s is unsupervised. `y` should be None." % self.__class__.__name__)

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, add_label_ids, del_label_ids) = self._convert_X(X_target, tokenized=tokenized, is_training=is_training)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)

            if is_training:
                data["add_label_ids"] = np.array(add_label_ids, dtype=np.int32)
                data["del_label_ids"] = np.array(del_label_ids, dtype=np.int32)

            # backup for answer mapping
            data[com.BACKUP_DATA + "input_tokens"] = input_tokens
            data[com.BACKUP_DATA + "tokenized"] = [tokenized]
            data[com.BACKUP_DATA + "X_target"] = X_target

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, tokenized, is_training):

        # tokenize input texts and scan over corpus
        input_tokens = []
        tokenized_input_ids = []
        vocab_size = len(self.tokenizer.vocab)
        vocab_ind = list(range(vocab_size))
        vocab_p = [0] * vocab_size
        for idx, sample in enumerate(X_target):
            _input_tokens = self._convert_x(sample, tokenized)

            # skip noise training data
            if is_training:
                if len(_input_tokens) == 0 or len(_input_tokens) > self.max_seq_length:
                    continue
            else:
                com.truncate_segments([_input_tokens], self.max_seq_length - 1, truncate_method=self.truncate_method)

            # count char
            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)
            if is_training:
                for _input_id in _input_ids:
                    vocab_p[_input_id] += 1
            _input_ids.insert(0, 1)

            input_tokens.append(_input_tokens)
            tokenized_input_ids.append(_input_ids)
        if is_training:
            vocab_p_sum = sum(vocab_p)
            vocab_p = [n / vocab_p_sum for n in vocab_p]

        input_ids = []
        add_label_ids = []
        del_label_ids = []
        for idx in range(len(tokenized_input_ids)):
            _input_ids = tokenized_input_ids[idx]

            nonpad_seq_length = len(_input_ids)
            for _ in range(self.max_seq_length - nonpad_seq_length):
                _input_ids.append(0)

            _add_label_ids = []
            _del_label_ids = []

            # add/del
            if is_training:
                if (idx + 1) % 10000 == 0:
                    tf.logging.info("Sampling wrong tokens of input %d" % (idx + 1))

                _add_label_ids = [0] * self.max_seq_length
                _del_label_ids = [0] * self.max_seq_length

                max_add = np.sum(np.random.random(nonpad_seq_length) < self._add_prob)
                max_del = np.sum(np.random.random(nonpad_seq_length) < self._del_prob)

                sample_wrong_tokens(
                    _input_ids, _add_label_ids, _del_label_ids,
                    max_add=max_add, max_del=max_del,
                    nonpad_seq_length=nonpad_seq_length,
                    vocab_size=vocab_size,
                    vocab_ind=vocab_ind,
                    vocab_p=vocab_p,
                )

            input_ids.append(_input_ids)
            add_label_ids.append(_add_label_ids)
            del_label_ids.append(_del_label_ids)

        return input_tokens, input_ids, add_label_ids, del_label_ids

    def _convert_x(self, x, tokenized):
        try:
            if not tokenized:
                # deal with general inputs
                if isinstance(x, str):
                    return self.tokenizer.tokenize(x)

            # deal with tokenized inputs
            elif isinstance(x[0], str):
                return x
        except Exception:
            raise ValueError("Wrong input format (%s)." % (x))

        # deal with tokenized and multiple inputs
        raise ValueError("%s only supports single sentence inputs." % self.__class__.__name__)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "input_ids"),
            "add_label_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "add_label_ids"),
            "del_label_ids": tf.placeholder(tf.int32, [None, self.max_seq_length], "del_label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        model = RecBERT(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            add_label_ids=placeholders["add_label_ids"],
            del_label_ids=placeholders["del_label_ids"],
            sample_weight=placeholders.get("sample_weight"),
            add_prob=self._add_prob,
            del_prob=self._del_prob,
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [
            self.tensors["add_preds"],
            self.tensors["del_preds"],
            self.tensors["add_loss"],
            self.tensors["del_loss"],
        ]
        if from_tfrecords:
            ops.extend([
                self.placeholders["input_ids"],
                self.placeholders["add_label_ids"],
                self.placeholders["del_label_ids"],
            ])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_inputs = output_arrays[-3]
            batch_add_labels = output_arrays[-2]
            batch_del_labels = output_arrays[-1]
        else:
            batch_inputs = feed_dict[self.placeholders["input_ids"]]
            batch_add_labels = feed_dict[self.placeholders["add_label_ids"]]
            batch_del_labels = feed_dict[self.placeholders["del_label_ids"]]
        batch_mask = (batch_inputs != 0)

        # add accuracy
        batch_add_preds = output_arrays[0]
        add_accuracy = np.sum((batch_add_preds == batch_add_labels) * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # del accuracy
        batch_del_preds = output_arrays[1]
        del_accuracy = np.sum((batch_del_preds == batch_del_labels) * batch_mask) / (np.sum(batch_mask) + 1e-6)

        # add loss
        batch_add_losses = output_arrays[2]
        add_loss = np.mean(batch_add_losses)

        # del loss
        batch_del_losses = output_arrays[3]
        del_loss = np.mean(batch_del_losses)

        info = ""
        if self._add_prob > 0:
            info += ", add_accuracy %.4f" % add_accuracy
            info += ", add_loss %.6f" % add_loss
        if self._del_prob > 0:
            info += ", del_accuracy %.4f" % del_accuracy
            info += ", del_loss %.6f" % del_loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["add_preds"], self.tensors["del_preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):
        input_ids = self.data["input_ids"]

        # integrated preds
        preds = []
        add_preds = com.transform(output_arrays[0], n_inputs)
        del_preds = com.transform(output_arrays[1], n_inputs)
        input_tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        for idx in range(n_inputs):
            _input_tokens = [""] + input_tokens[idx]
            _input_length = np.sum(input_ids[idx] > 0)
            _add_preds = add_preds[idx]
            _del_preds = del_preds[idx]

            if tokenized:
                n = 0
                _output_tokens = [token for token in _input_tokens]
                for i in range(_input_length):
                    if self._del_prob > 0 and _del_preds[i] != 0 and i > 0:
                        _output_tokens[i + n] = "{del:%s}" % _output_tokens[i + n]
                    if self._add_prob > 0 and _add_preds[i] != 0:
                        _token = "{add:%s}" % self.tokenizer.convert_ids_to_tokens([_add_preds[i]])[0]
                        _output_tokens.insert(i + 1 + n, _token)
                        n += 1
                preds.append(_output_tokens[1:])
            else:
                _text = text[idx]
                _mapping_start, _mapping_end = com.align_tokens_with_text(_input_tokens, _text, self._do_lower_case)

                n = 0
                for i in range(_input_length):
                    if self._del_prob > 0 and _del_preds[i] != 0 and i > 0:
                        _start_ptr = _mapping_start[i] + n
                        _end_ptr = _mapping_end[i] + n
                        _del_token = _text[_start_ptr: _end_ptr]

                        _token = "{del:%s}" % _del_token
                        _text = _text[:_start_ptr] + _token + _text[_end_ptr:]
                        n += len(_token) - len(_del_token)
                    if self._add_prob > 0 and _add_preds[i] != 0:
                        _token = self.tokenizer.convert_ids_to_tokens([_add_preds[i]])[0]
                        _token = "{add:%s}" % _token
                        _ptr = _mapping_end[i] + n
                        _text = _text[:_ptr] + _token + _text[_ptr:]
                        n += len(_token)
                preds.append(_text)

        outputs = {}
        outputs["preds"] = preds

        return outputs
