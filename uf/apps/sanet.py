import numpy as np

from .base import MRCModule
from .bert import BERTMRC
from ..third import tf
from ..model.bert import BERTEncoder, BERTConfig
from ..model.albert import ALBERTEncoder, ALBERTConfig
from ..model.sanet import SANetDecoder
from ..token import WordPieceTokenizer
from .. import com


class SANetMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on SANet. """
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
        reading_module="bert",
        split_signs=",，。?？!！;；",
        alpha=0.5,
        truncate_method="longer-FO",
    ):
        self.__init_args__ = locals()
        super(MRCModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self.split_signs = list(map(str, split_signs))
        self._do_lower_case = do_lower_case
        self._on_predict = False
        self._reading_module = reading_module
        self._alpha = alpha

        if reading_module == "albert":
            self.bert_config = ALBERTConfig.from_json_file(config_file)
        else:
            self.bert_config = BERTConfig.from_json_file(config_file)

        assert reading_module in ("bert", "albert", "electra"), (
            "Invalid value of `reading_module`: %s. Pick one from "
            "`bert`, `albert` and `electra`."
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
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            X_target = X_tokenized if tokenized else X
            (input_tokens, input_ids, input_mask, sa_mask, segment_ids,
             doc_ids, doc_text, doc_start) = self._convert_X(X_target, tokenized=tokenized)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            data["input_mask"] = np.array(input_mask, dtype=np.int32)
            data["sa_mask"] = np.array(sa_mask, dtype=np.int32)
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
        sa_mask = []
        segment_ids = []
        doc_ids = []
        doc_text = []
        doc_start = []
        for idx, segments in enumerate(segment_input_tokens):
            _input_tokens = ["[CLS]"]
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _sa_mask = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.int32)
            _sa_mask[0, 0] = 1

            _doc_sent_tokens = segments.pop("doc")
            _doc_sent_num = len(_doc_sent_tokens)
            segments = list(segments.values()) + _doc_sent_tokens
            com.truncate_segments(segments, self.max_seq_length - (len(segments) - _doc_sent_num + 1) - 1, truncate_method=self.truncate_method)

            # split doc and other infos after truncation
            non_doc_segments = segments[:-_doc_sent_num]
            doc_segments = segments[-_doc_sent_num:]

            # non-doc
            for s_id, segment in enumerate(non_doc_segments):
                _segment_len = len(segment) + 1    # [SEP]
                _start_pos = len(_input_tokens)
                _end_pos = _start_pos + len(segment)
                _sa_mask[_start_pos: _end_pos, _start_pos: _end_pos] = 1
                _sa_mask[_end_pos, _end_pos] = 1    # [SEP] pay attention to itself
                _input_tokens.extend(segment + ["[SEP]"])
                _input_mask.extend([1] * _segment_len)
                _segment_ids.extend([min(s_id, 1)] * _segment_len)

            # doc
            _doc_start = len(_input_tokens)
            for s_id, segment in enumerate(doc_segments):
                _segment_len = len(segment)
                _start_pos = len(_input_tokens)
                _end_pos = _start_pos + _segment_len
                _sa_mask[_start_pos: _end_pos, _start_pos: _end_pos] = 1
                _input_tokens.extend(segment)
                _input_mask.extend([1] * _segment_len)
                _segment_ids.extend([1] * _segment_len)
            _input_tokens.append("[SEP]")
            _input_mask.append(1)
            _segment_ids.append(1)

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
            sa_mask.append(np.reshape(_sa_mask, [-1]).tolist())
            segment_ids.append(_segment_ids)
            doc_ids.append(_doc_ids)
            doc_text.append(X_target[idx]["doc"])
            doc_start.append(_doc_start)

        return (input_tokens, input_ids, input_mask, sa_mask, segment_ids, doc_ids, doc_text, doc_start)

    def _convert_x(self, x, tokenized):
        output = {}

        if not isinstance(x, dict) or "doc" not in x:
            raise ValueError(
                "Wrong input format of `y`. An untokenized example: "
                "`y = [{\"doc\": \"...\", \"question\": \"...\", ...}, "
                "None, ...]`"
            )

        for key in x:
            if not tokenized:
                chars = self.tokenizer.tokenize(x[key])

                # deal with general inputs
                if key == "doc":
                    sents = []
                    last = []
                    for char in chars:
                        last.append(char)
                        if char in self.split_signs:
                            sents.append(last)
                            last = []
                    if last:
                        sents.append(last)
                    output[key] = sents
                    continue
                output[key] = chars
                continue

            # deal with tokenized inputs
            output[key] = x[key]

        return output

    def _set_placeholders(self, target, on_export=False, **kwargs):
        self.placeholders = {
            "input_ids": com.get_placeholder(target, "input_ids", [None, self.max_seq_length], tf.int32),
            "input_mask": com.get_placeholder(target, "input_mask", [None, self.max_seq_length], tf.int32),
            "sa_mask": com.get_placeholder(target, "sa_mask", [None, self.max_seq_length ** 2], tf.int32),
            "segment_ids": com.get_placeholder(target, "segment_ids", [None, self.max_seq_length], tf.int32),
            "label_ids": com.get_placeholder(target, "label_ids", [None, 2], tf.int32),
            "has_answer": com.get_placeholder(target, "has_answer", [None], tf.int32),
        }
        if not on_export:
            self.placeholders["sample_weight"] = com.get_placeholder(target, "sample_weight", [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        def _get_encoder(model_name):
            if model_name == "bert":
                encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders["input_ids"],
                    input_mask=split_placeholders["input_mask"],
                    segment_ids=split_placeholders["segment_ids"],
                    **kwargs,
                )
            elif model_name == "albert":
                encoder = ALBERTEncoder(
                    albert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders["input_ids"],
                    input_mask=split_placeholders["input_mask"],
                    segment_ids=split_placeholders["segment_ids"],
                    drop_pooler=self._drop_pooler,
                    **kwargs,
                )
            elif model_name == "electra":
                encoder = BERTEncoder(
                    bert_config=self.bert_config,
                    is_training=is_training,
                    input_ids=split_placeholders["input_ids"],
                    input_mask=split_placeholders["input_mask"],
                    segment_ids=split_placeholders["segment_ids"],
                    scope="electra",
                    **kwargs,
                )
            return encoder

        encoder = _get_encoder(self._reading_module)
        decoder = SANetDecoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_tensor=encoder.get_sequence_output(),
            sa_mask=split_placeholders["sa_mask"],
            label_ids=split_placeholders["label_ids"],
            sample_weight=split_placeholders.get("sample_weight"),
            alpha=self._alpha,
            trainable=True,
            **kwargs,
        )
        return decoder.get_forward_outputs()


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/embeddings": num_hidden_layers + 2,
        "/sentence_attention/": 1,
        "/cls/mrc/": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/layer_%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power
