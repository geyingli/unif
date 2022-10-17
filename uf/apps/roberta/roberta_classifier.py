from ..base.base_classifier import ClassifierModule
from ..bert.bert_classifier import BERTClassifier


class RoBERTaClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on RoBERTa. """
    
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES
