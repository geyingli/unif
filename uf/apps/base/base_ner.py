import numpy as np

from ...core import BaseModule


class NERModule(BaseModule):
    """ Application class of name entity recognition (NER). """

    O_ID = 0
    B_ID = 1
    I_ID = 2
    E_ID = 3
    S_ID = 4

    def _get_cascade_f1(self, preds, labels, mask):
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

        pre_token = tp_token / (tp_token + fp_token + 1e-6)
        rec_token = tp_token / (tp_token + fn_token + 1e-6)
        f1_token = 2 * pre_token * rec_token / (pre_token + rec_token + 1e-6)

        pre_entity = tp_entity / (tp_entity + fp_entity + 1e-6)
        rec_entity = tp_entity / (tp_entity + fn_entity + 1e-6)
        f1_entity = 2 * pre_entity * rec_entity / (pre_entity + rec_entity + 1e-6)

        return f1_token, f1_entity

    def _get_entities(self, ids, B_id=None):
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
