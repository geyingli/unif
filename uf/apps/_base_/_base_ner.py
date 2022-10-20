import numpy as np

from ...core import BaseModule
from ...third import tf
from ... import com


class NERModule(BaseModule):
    """ Application class of name entity recognition (NER). """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    O_ID = 0
    B_ID = 1
    I_ID = 2
    E_ID = 3
    S_ID = 4

    def _get_cascade_f1(self, preds, labels, mask):
        """ Micro and macro F1. """
        metrics = {}

        # f1 of each type
        B_ids = []
        for i, entity_type in enumerate(self.entity_types):
            B_id = 1 + i * 4
            B_ids.append(B_id)
            f1_token, f1_entity = self._get_f1(preds, labels, mask, B_id=B_id)
            metrics["f1/%s-token" % entity_type] = f1_token
            metrics["f1/%s-entity" % entity_type] = f1_entity

        # macro f1
        f1_macro_token = np.mean([metrics[key] for key in metrics if "-token" in key])
        f1_macro_entity = np.mean([metrics[key] for key in metrics if "-entity" in key])
        metrics["macro f1/token"] = f1_macro_token
        metrics["macro f1/entity"] = f1_macro_entity

        # micro f1
        f1_micro_token, f1_micro_entity = self._get_f1(preds, labels, mask, B_id=B_ids)
        metrics["micro f1/token"] = f1_micro_token
        metrics["micro f1/entity"] = f1_micro_entity

        return metrics

    def _get_f1(self, preds, labels, mask, B_id=None):
        """ F1. """

        tp_token, fp_token, fn_token = 0, 0, 0
        tp_entity, fp_entity, fn_entity = 0, 0, 0
        for _preds, _labels, _mask in zip(preds, labels, mask):
            length = int(np.sum(_mask))

            # entity metrics
            _entities_pred = self._get_entities(_preds[:length], B_id)
            _entities_label = self._get_entities(_labels[:length], B_id)
            for _entity_pred in _entities_pred:
                if _entity_pred in _entities_label:
                    tp_entity += 1
                else:
                    fp_entity += 1
            for _entity_label in _entities_label:
                if _entity_label not in _entities_pred:
                    fn_entity += 1

            # token metrics
            _preds = np.zeros((length))
            _labels = np.zeros((length))
            for _entity in _entities_pred:
                _preds[_entity[0]:_entity[1] + 1] = 1
            for _entity in _entities_label:
                _labels[_entity[0]:_entity[1] + 1] = 1
            for _pred_id, _label_id in zip(_preds, _labels):
                if _pred_id == 1:
                    if _label_id == 1:
                        tp_token += 1
                    else:
                        fp_token += 1
                elif _label_id == 1:
                    fn_token += 1

        pre_token = tp_token / max(tp_token + fp_token, 1)
        rec_token = tp_token / max(tp_token + fn_token, 1)
        f1_token = 2 * pre_token * rec_token / max(pre_token + rec_token, 1)

        pre_entity = tp_entity / max(tp_entity + fp_entity, 1)
        rec_entity = tp_entity / max(tp_entity + fn_entity, 1)
        f1_entity = 2 * pre_entity * rec_entity / max(pre_entity + rec_entity, 1)

        return f1_token, f1_entity

    def _get_entities(self, ids, B_id=None):
        """ Parse BIESO tags. """
        if not B_id:
            B_id = self.B_ID
            I_id = self.I_ID
            E_id = self.E_ID
            S_id = self.S_ID
            BIES = [(B_id, I_id, E_id, S_id)]
        elif isinstance(B_id, list):
            BIES = []
            for _B_id in B_id:
                I_id = _B_id + 1
                E_id = _B_id + 2
                S_id = _B_id + 3
                BIES += [(_B_id, I_id, E_id, S_id)]
        else:
            I_id = B_id + 1
            E_id = B_id + 2
            S_id = B_id + 3
            BIES = [(B_id, I_id, E_id, S_id)]

        entities = []
        for k in range(len(BIES)):
            (B_id, I_id, E_id, S_id) = BIES[k]
            on_entity = False
            start_id = 0
            for i in range(len(ids)):
                if on_entity:
                    if ids[i] == I_id:
                        pass
                    elif ids[i] == E_id:
                        on_entity = False
                        entities.append([start_id, i])
                    elif ids[i] == S_id:
                        on_entity = False
                        entities.append([i, i])
                    else:
                        on_entity = False
                else:
                    if ids[i] == B_id:
                        on_entity = True
                        start_id = i
                    elif ids[i] == S_id:
                        entities.append([i, i])
        return entities

    def _convert_x(self, x, tokenized):
        """ Convert text sample. """

        # deal with untokenized inputs
        if not tokenized:

            # deal with general inputs
            if isinstance(x, str):
                return [self.tokenizer.tokenize(x)]

            # deal with multiple inputs
            return [self.tokenizer.tokenize(seg) for seg in x]

        # deal with tokenized inputs
        if isinstance(x[0], str):
            return [x]

        # deal with tokenized and multiple inputs
        return x

    def _convert_y(self, y, input_ids, tokenized=False):
        label_ids = []

        for idx, (_y, _input_ids) in enumerate(zip(y, input_ids)):
            if not _y:
                label_ids.append([self.O_ID] * self.max_seq_length)
                continue

            if isinstance(_y, str):
                _entity_tokens = self.tokenizer.tokenize(_y)
                _entity_ids = [self.tokenizer.convert_tokens_to_ids(_entity_tokens)]
            elif isinstance(_y, list):
                if isinstance(_y[0], str):
                    if tokenized:
                        _entity_ids = [self.tokenizer.convert_tokens_to_ids(_y)]
                    else:
                        _entity_ids = []
                        for _entity in _y:
                            _entity_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_entity)))
                elif isinstance(_y[0], list):
                    _entity_ids = []
                    for _entity in _y:
                        _entity_ids.append(self.tokenizer.convert_tokens_to_ids(_entity))
            else:
                raise ValueError("`y` should be a list of entity strings.")

            # tagging
            _label_ids = [self.O_ID for _ in range(self.max_seq_length)]
            for _entity in _entity_ids:
                start_positions = com.find_all_boyer_moore(_input_ids, _entity)
                if not start_positions:
                    tf.logging.warning(
                        "Failed to find the mapping of entity to "
                        "inputs at line %d. A possible reason is "
                        "that the entity span is truncated due "
                        "to the `max_seq_length` setting." % (idx)
                    )
                    continue

                for start_position in start_positions:
                    end_position = start_position + len(_entity) - 1
                    if start_position == end_position:
                        _label_ids[start_position] = self.S_ID
                    else:
                        for i in range(start_position, end_position + 1):
                            _label_ids[i] = self.I_ID
                        _label_ids[start_position] = self.B_ID
                        _label_ids[end_position] = self.E_ID

            label_ids.append(_label_ids)

        return label_ids

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["losses"]]
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
        batch_preds = output_arrays[0]
        f1_token, f1_entity = self._get_f1(batch_preds, batch_labels, batch_mask)

        # loss
        batch_losses = output_arrays[1]
        loss = np.mean(batch_losses)

        info = ""
        info += ", f1/token %.4f" % f1_token
        info += ", f1/entity %.4f" % f1_entity
        info += ", loss %.6f" % loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["probs"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # probs
        probs = com.transform(output_arrays[0], n_inputs)

        # preds
        all_preds = np.argmax(probs, axis=-1)
        tokens = self.data[com.BACKUP_DATA + "input_tokens"]
        mask = self.data["input_mask"]
        text = self.data[com.BACKUP_DATA + "X_target"]
        tokenized = self.data[com.BACKUP_DATA + "tokenized"][0]
        preds = []
        for i in range(len(all_preds)):
            _preds = all_preds[i]
            _tokens = tokens[i]
            _mask = mask[i]
            _text = text[i]

            _input_length = int(np.sum(_mask))
            _entities = self._get_entities(_preds[:_input_length])
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

        outputs = {}
        outputs["preds"] = preds
        outputs["probs"] = probs

        return outputs

    def _get_score_ops(self):
        return [self.tensors["preds"], self.tensors["losses"]]

    def _get_score_outputs(self, output_arrays, n_inputs):

        # f1
        preds = com.transform(output_arrays[0], n_inputs)
        labels = self.data["label_ids"]
        mask = self.data["input_mask"]
        f1_token, f1_entity = self._get_f1(preds, labels, mask)

        # loss
        losses = com.transform(output_arrays[1], n_inputs)
        loss = np.mean(losses)

        outputs = {}
        outputs["f1/token"] = f1_token
        outputs["f1/entity"] = f1_entity
        outputs["loss"] = loss

        return outputs

