from .._base_._base_classifier import ClassifierModule
from ..bert.bert_classifier import BERTClassifier


class RoBERTaClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on RoBERTa. """
    pass
