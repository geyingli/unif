

from ..com import unimported_module

from .bert import BERTLM
from .roberta import RoBERTaLM
from .albert import ALBERTLM
from .electra import ELECTRALM
from .dilated import DilatedLM
from .recbert import RecBERTLM
from .vae import VAELM
from .spe import SPELM
from .gpt2 import GPT2LM
from .unilm import UniLM
from .textcnn import TextCNNClassifier
from .bert import BERTClassifier
from .roberta import RoBERTaClassifier
from .albert import ALBERTClassifier
from .electra import ELECTRAClassifier
from .widedeep import WideAndDeepClassifier
from .sembert import SemBERTClassifier
from .performer import PerformerClassifier
from .uda import UDAClassifier
from .tinybert import TinyBERTClassifier
from .tinybert import TinyBERTBinaryClassifier
from .fastbert import FastBERTClassifier
from .stockbert import StockBERTClassifier
from .bert import BERTBinaryClassifier
from .roberta import RoBERTaBinaryClassifier
from .albert import ALBERTBinaryClassifier
from .electra import ELECTRABinaryClassifier
from .bert import BERTSeqClassifier
from .roberta import RoBERTaSeqClassifier
from .albert import ALBERTSeqClassifier
from .electra import ELECTRASeqClassifier
from .bert import BERTSeqMultiTaskClassifier
from .widedeep import WideAndDeepRegressor
from .bert import BERTNER
from .bert import BERTCRFNER
from .bert import BERTCRFCascadeNER
from .bert import BERTMRC
from .bert import BERTVerifierMRC
from .roberta import RoBERTaMRC
from .albert import ALBERTMRC
from .electra import ELECTRAMRC
from .retroreader import RetroReaderMRC
from .sanet import SANetMRC
from .transformer import TransformerMT


# sentencepiece==0.1.85
try:
    from .xlnet import XLNetClassifier
    from .xlnet import XLNetBinaryClassifier
except ModuleNotFoundError:
    XLNetClassifier = unimported_module("XLNetClassifier", "sentencepiece")
    XLNetBinaryClassifier = unimported_module("XLNetBinaryClassifier", "sentencepiece")

del unimported_module


__all__ = [
    "BERTLM",
    "RoBERTaLM",
    "ALBERTLM",
    "ELECTRALM",
    "VAELM",
    "GPT2LM",
    "UniLM",
    "TextCNNClassifier",
    "BERTClassifier",
    "XLNetClassifier",
    "RoBERTaClassifier",
    "ALBERTClassifier",
    "ELECTRAClassifier",
    "WideAndDeepClassifier",
    "SemBERTClassifier",
    "UDAClassifier",
    "TinyBERTClassifier",
    "TinyBERTBinaryClassifier",
    "FastBERTClassifier",
    "BERTBinaryClassifier",
    "XLNetBinaryClassifier",
    "RoBERTaBinaryClassifier",
    "ALBERTBinaryClassifier",
    "ELECTRABinaryClassifier",
    "BERTSeqClassifier",
    "RoBERTaSeqClassifier",
    "ALBERTSeqClassifier",
    "ELECTRASeqClassifier",
    "BERTSeqMultiTaskClassifier",
    "WideAndDeepRegressor",
    "BERTNER",
    "BERTCRFNER",
    "BERTCRFCascadeNER",
    "BERTMRC",
    "BERTVerifierMRC",
    "RoBERTaMRC",
    "ALBERTMRC",
    "ELECTRAMRC",
    "RetroReaderMRC",
    "SANetMRC",
    "TransformerMT",

    # trial
    "DilatedLM",
    "RecBERTLM",
    "SPELM",
    "StockBERTClassifier",
    "PerformerClassifier",
]