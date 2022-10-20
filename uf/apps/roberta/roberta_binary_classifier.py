from .._base_._base_binary_classifier import BinaryClassifierModule
from ..bert.bert_binary_classifier import BERTBinaryClassifier


class RoBERTaBinaryClassifier(BERTBinaryClassifier, BinaryClassifierModule):
    """ Multi-label classifier on RoBERTa. """
    pass