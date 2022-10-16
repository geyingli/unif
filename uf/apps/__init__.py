from ..com import unimported_module

from .bert.bert_lm import BERTLM
from .roberta.roberta_lm import RoBERTaLM
from .albert.albert_lm import ALBERTLM
from .electra.electra_lm import ELECTRALM
from .dilated.dilated_lm import DilatedLM
from .recbert.recbert_lm import RecBERTLM
from .vae.vae_lm import VAELM
from .spe.spe_lm import SPELM
from .gpt2.gpt2_lm import GPT2LM
from .unilm.unilm_lm import UniLM
from .textcnn.textcnn_classifier import TextCNNClassifier
from .bert.bert_classifier import BERTClassifier
from .roberta.roberta_classifier import RoBERTaClassifier
from .albert.albert_classifier import ALBERTClassifier
from .electra.electra_classifier import ELECTRAClassifier
from .widedeep.widedeep_classifier import WideDeepClassifier
from .sembert.sembert_classifier import SemBERTClassifier
from .performer.performer_classifier import PerformerClassifier
from .uda.uda_classifier import UDAClassifier
from .tinybert.tinybert_classifier import TinyBERTClassifier
from .tinybert.tinybert_binary_classifier import TinyBERTBinaryClassifier
from .fastbert.fastbert_classifier import FastBERTClassifier
from .stockbert.stockbert_classifier import StockBERTClassifier
from .bert.bert_binary_classifier import BERTBinaryClassifier
from .roberta.roberta_binary_classifier import RoBERTaBinaryClassifier
from .albert.albert_binary_classifier import ALBERTBinaryClassifier
from .electra.electra_binary_classifier import ELECTRABinaryClassifier
from .bert.bert_seq_classifier import BERTSeqClassifier
from .roberta.roberta_seq_classifier import RoBERTaSeqClassifier
from .albert.albert_seq_classifier import ALBERTSeqClassifier
from .electra.electra_seq_classifier import ELECTRASeqClassifier
from .bert.bert_seq_cross_classifier import BERTSeqCrossClassifier
from .widedeep.widedeep_regressor import WideDeepRegressor
from .bert.bert_ner import BERTNER
from .bert.bert_crf_ner import BERTCRFNER
from .bert.bert_crf_cascade_ner import BERTCRFCascadeNER
from .bert.bert_mrc import BERTMRC
from .bert.bert_verifier_mrc import BERTVerifierMRC
from .roberta.roberta_mrc import RoBERTaMRC
from .albert.albert_mrc import ALBERTMRC
from .electra.electra_mrc import ELECTRAMRC
from .retroreader.retroreader_mrc import RetroReaderMRC
from .sanet.sanet_mrc import SANetMRC
from .transformer.transformer_mt import TransformerMT


try:
    from .xlnet.xlnet_classifier import XLNetClassifier
    from .xlnet.xlnet_binary_classifier import XLNetBinaryClassifier
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
    "WideDeepClassifier",
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
    "BERTSeqCrossClassifier",
    "WideDeepRegressor",
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
