import numpy as np

from .bert import BERTEncoder
from .bert_ner import BERTNER
from .._base_._base_ner import NERModule
from ..crf.crf import CRFDecoder, viterbi_decode
from ... import com


class BERTCRFNER(BERTNER, NERModule):
    """ Named entity recognization on BERT with CRF. """

    def _forward(self, is_training, placeholders, **kwargs):

        encoder = BERTEncoder(
            bert_config=self.bert_config,
            is_training=is_training,
            input_ids=placeholders["input_ids"],
            input_mask=placeholders["input_mask"],
            segment_ids=placeholders["segment_ids"],
            **kwargs,
        )
        encoder_output = encoder.get_sequence_output()
        decoder = CRFDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=placeholders["input_mask"],
            label_ids=placeholders["label_ids"],
            label_size=5,
            sample_weight=placeholders.get("sample_weight"),
            scope="cls/sequence",
            **kwargs,
        )
        return decoder.get_forward_outputs()

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["input_mask"], self.placeholders["label_ids"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_mask = output_arrays[-2]
            batch_labels = output_arrays[-1]
        else:
            batch_mask = feed_dict[self.placeholders["input_mask"]]
            batch_labels = feed_dict[self.placeholders["label_ids"]]

        # f1
        batch_logits = output_arrays[0]
        batch_transition_matrix = output_arrays[1]
        batch_input_length = np.sum(batch_mask, axis=-1)
        batch_preds = []
        for logit, seq_len in zip(batch_logits, batch_input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], batch_transition_matrix)
            batch_preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[2]
        loss = np.mean(batch_losses)

        info = ""
        info += ", f1/token %.4f" % f1_token
        info += ", f1/entity %.4f" % f1_entity
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # preds
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(logits)):
            _logits = logits[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _viterbi_seq, _ = viterbi_decode(_logits[:_input_length], transition_matrix)
            _entities = self._get_entities(_viterbi_seq)
            _preds = []
            if not _entities:
                preds.append(_preds)
                continue

            if not tokenized:
                if isinstance(_text, list):
                    _text = " ".join(_text)
                _mapping_start, _mapping_end = com.align_tokens_with_text(_tokens, _text, self._do_lower_case)

            for _entity in _entities:
                _start, _end = _entity[0], _entity[1]
                if tokenized:
                    _entity_tokens = _tokens[_start: _end + 1]
                    _preds.append(_entity_tokens)
                else:
                    try:
                        _text_start = _mapping_start[_start]
                        _text_end = _mapping_end[_end]
                    except Exception:
                        continue
                    _entity_text = _text[_text_start: _text_end]
                    _preds.append(_entity_text)
            preds.append(_preds)

        # probs
        probs = logits

        outputs = {}
        outputs["preds"] = preds
        outputs["logits"] = probs

        return outputs

    def _get_score_ops(self):
        return [self._tensors["logits"], self._tensors["transition_matrix"], self._tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # f1
        logits = com.transform(output_arrays[0], n_inputs)
        transition_matrix = output_arrays[1][0]
        mask = self.data["input_mask"]
        labels = self.data["label_ids"]
        input_length = np.sum(mask, axis=-1)
        preds = []
        for logit, seq_len in zip(logits, input_length):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_matrix)
            preds.append(viterbi_seq)
        f1_token, f1_entity = self._get_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[2], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["f1/token"] = f1_token
        outputs["f1/entity"] = f1_entity
        outputs["loss"] = loss

        return outputs
