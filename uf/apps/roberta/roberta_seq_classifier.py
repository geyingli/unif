from .._base_._base_classifier import ClassifierModule
from ..bert.bert_seq_classifier import BERTSeqClassifier


class RoBERTaSeqClassifier(BERTSeqClassifier, ClassifierModule):
    """ Sequence labeling classifier on RoBERTa. """
    pass
