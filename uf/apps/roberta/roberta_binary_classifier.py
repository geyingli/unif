from .._base_._base_classifier import ClassifierModule
from ..bert.bert_binary_classifier import BERTBinaryClassifier


class RoBERTaBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    """ Multi-label classifier on RoBERTa. """
    pass