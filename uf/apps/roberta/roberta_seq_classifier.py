from .._base_._base_seq_classifier import SeqClassifierModule
from ..bert.bert_seq_classifier import BERTSeqClassifier


class RoBERTaSeqClassifier(BERTSeqClassifier, SeqClassifierModule):
    """ Sequence labeling classifier on RoBERTa. """
    pass
