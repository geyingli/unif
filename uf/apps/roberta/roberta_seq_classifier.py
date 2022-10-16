from ..base.base_classifier import ClassifierModule
from ..bert.bert_seq_classifier import BERTSeqClassifier


class RoBERTaSeqClassifier(BERTSeqClassifier, ClassifierModule):
    """ Sequence labeling classifier on RoBERTa. """
    _INFER_ATTRIBUTES = BERTSeqClassifier._INFER_ATTRIBUTES
