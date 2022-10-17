from ..base.base_mrc import MRCModule
from ..bert.bert_mrc import BERTMRC


class RoBERTaMRC(BERTMRC, MRCModule):
    """ Machine reading comprehension on RoBERTa. """
    
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES
