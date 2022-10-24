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
from .rnn.rnn_classifier import RNNClassifier
from .rnn.bi_rnn_classifier import BiRNNClassifier
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
from .adabert.adabert_classifier import AdaBERTClassifier
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
from .chatbot.chatbot_mt import ChatbotMT
try:
    from .xlnet.xlnet_classifier import XLNetClassifier
    from .xlnet.xlnet_binary_classifier import XLNetBinaryClassifier
except (ModuleNotFoundError, ImportError):
    XLNetClassifier = unimported_module(
        "XLNetClassifier",
        "Module `sentencepiece` is required to launch XLNetClassifier. "
        "Try `pip install sentencepiece` or build from source."
    )
    XLNetBinaryClassifier = unimported_module(
        "XLNetBinaryClassifier",
        "Module `sentencepiece` is required to launch XLNetBinaryClassifier. "
        "Try `pip install sentencepiece` or build from source."
    )
try:
    from .nasnet.pnasnet_classifier import PNasNetClassifier
except (ModuleNotFoundError, ImportError):
    PNasNetClassifier = unimported_module(
        "PNasNetClassifier",
        "Module `tf_slim` is required to launch PNasNetClassifier. "
        "Try `pip install tf_slim` or build from source."
    )

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
    "RNNClassifier",
    "BiRNNClassifier",
    "BERTClassifier",
    "XLNetClassifier",
    "RoBERTaClassifier",
    "ALBERTClassifier",
    "ELECTRAClassifier",
    "WideDeepClassifier",
    "SemBERTClassifier",
    "UDAClassifier",
    "PerformerClassifier",
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
    "PNasNetClassifier",

    # trial
    "DilatedLM",
    "RecBERTLM",
    "SPELM",
    "StockBERTClassifier",
    "AdaBERTClassifier",
    "ChatbotMT",
]
