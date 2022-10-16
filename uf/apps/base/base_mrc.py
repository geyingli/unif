from ...core import BaseModule


class MRCModule(BaseModule):
    """ Application class of machine reading comprehension (MRC). """

    def _get_em_and_f1(self, preds, labels):
        em, f1 = 0, 0
        for _preds, _labels in zip(preds, labels):
            start_pred, end_pred = int(_preds[0]), int(_preds[1])
            start_label, end_label = int(_labels[0]), int(_labels[1])

            # no answer prediction
            if start_pred == 0 or end_pred == 0 or start_pred > end_pred:
                if start_label == 0:
                    em += 1
                    f1 += 1

            # answer prediction (no intersection)
            elif start_pred > end_label or end_pred < start_label:
                pass

            # answer prediction (has intersection)
            else:
                tp = (min(end_pred, end_label) + 1 - max(start_pred, start_label))
                fp = (max(0, end_pred - end_label) + max(0, start_label - start_pred))
                fn = (max(0, start_pred - start_label) + max(0, end_label - end_pred))
                if fp + fn == 0:
                    em += 1
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f1 += 2 * precision * recall / (precision + recall + 1e-6)

        em /= len(labels)
        f1 /= len(labels)
        return em, f1
