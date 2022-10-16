from ..base.base_classifier import ClassifierModule
from ..bert.bert_binary_classifier import BERTBinaryClassifier


class RoBERTaBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    """ Multi-label classifier on RoBERTa. """
    _INFER_ATTRIBUTES = BERTBinaryClassifier._INFER_ATTRIBUTES
