import numpy as np

from .xlnet import XLNetEncoder, XLNetConfig, get_decay_power, SEG_ID_CLS, SEG_ID_PAD, CLS_ID, SEP_ID
from .._base_._base_binary_classifier import BinaryClsDecoder, BinaryClassifierModule
from ..bert.bert_binary_classifier import BERTBinaryClassifier
from ...token import SentencePieceTokenizer
from ...third import tf
from ... import com


class XLNetBinaryClassifier(BERTBinaryClassifier, BinaryClassifierModule):
    """ Multi-label classifier on XLNet. """

    def __init__(
        self,
        config_file,
        spm_file,
        max_seq_length=128,
        label_size=None,
        label_weight=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        do_lower_case=True,
        truncate_method="LIFO",
    ):
        self.__init_args__ = locals()
        super(BinaryClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.label_weight = label_weight
        self.truncate_method = truncate_method

        self.xlnet_config = XLNetConfig(json_path=config_file)
        self.tokenizer = SentencePieceTokenizer(spm_file, do_lower_case)
        self.decay_power = get_decay_power(self.xlnet_config.n_layer)

    def _convert_X(self, X_target, tokenized):

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
        for idx, segments in enumerate(segment_input_tokens):
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            com.truncate_segments(segments, self.max_seq_length - len(segments) - 1, truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_ids.extend(self.tokenizer.convert_tokens_to_ids(segment) + [SEP_ID])
                _input_mask.extend([0] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids.append(CLS_ID)
            _input_mask.append(0)
            _segment_ids.append(SEG_ID_CLS)

            # padding
            if len(_input_ids) < self.max_seq_length:
                delta_len = self.max_seq_length - len(_input_ids)
                _input_ids = [0] * delta_len + _input_ids
                _input_mask = [1] * delta_len + _input_mask
                _segment_ids = [SEG_ID_PAD] * delta_len + _segment_ids

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _forward(self, is_training, placeholders, **kwargs):

        input_ids = tf.transpose(placeholders["input_ids"], [1, 0])
        input_mask = tf.transpose(placeholders["input_mask"], [1, 0])
        segment_ids = tf.transpose(placeholders["segment_ids"], [1, 0])

        encoder = XLNetEncoder(
            xlnet_config=self.xlnet_config,
            is_training=is_training,
            input_ids=input_ids,
            seg_ids=segment_ids,
            input_mask=input_mask,
            **kwargs,
        )
        encoder_output = encoder.get_pooled_output()
        decoder = BinaryClsDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=placeholders["label_ids"],
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
            label_weight=self.label_weight,
            scope="cls/seq_relationship",
            **kwargs,
        )
        return decoder.get_forward_outputs()
