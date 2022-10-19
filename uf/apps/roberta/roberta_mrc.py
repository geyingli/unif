from .._base_._base_mrc import MRCModule
from ..bert.bert_mrc import BERTMRC


class RoBERTaMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on RoBERTa. """
    pass
