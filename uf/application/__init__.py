# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from uf.utils import unimported_module

from .bert import BERTLM
from .roberta import RoBERTaLM
from .albert import ALBERTLM
from .electra import ELECTRALM
from .gpt2 import GPT2LM
from .text_cnn import TextCNNClassifier
from .bert import BERTClassifier
from .roberta import RoBERTaClassifier
from .albert import ALBERTClassifier
from .electra import ELECTRAClassifier
from .performer import PerformerClassifier
from .tiny_bert import TinyBERTClassifier
from .fast_bert import FastBERTClassifier
from .bert import BERTBinaryClassifier
from .roberta import RoBERTaBinaryClassifier
from .albert import ALBERTBinaryClassifier
from .electra import ELECTRABinaryClassifier
from .bert import BERTSeqClassifier
from .roberta import RoBERTaSeqClassifier
from .albert import ALBERTSeqClassifier
from .electra import ELECTRASeqClassifier
from .bert import BERTNER
from .bert import BERTCRFNER
from .bert import BERTCRFCascadeNER
from .bert import BERTMRC
from .bert import BERTVerifierMRC
from .roberta import RoBERTaMRC
from .albert import ALBERTMRC
from .electra import ELECTRAMRC
from .retro_reader import RetroReaderMRC
from .sanet import SANetMRC
from .transformer import TransformerMT


# sentencepiece==0.1.85
try:
    from .xlnet import XLNetClassifier
    from .xlnet import XLNetBinaryClassifier
except ModuleNotFoundError:
    XLNetClassifier = unimported_module(
        'XLNetClassifier', 'sentencepiece')
    XLNetBinaryClassifier = unimported_module(
        'XLNetBinaryClassifier', 'sentencepiece')

# pyemd
try:
    from .bert_emd import BERTEMDClassifier
except ModuleNotFoundError:
    BERTEMDClassifier = unimported_module(
        'BERTEMDClassifier', 'pyemd')

del unimported_module


__all__ = [
    'BERTLM',
    'RoBERTaLM',
    'ALBERTLM',
    'ELECTRALM',
    'GPT2LM',
    'TextCNNClassifier',
    'BERTClassifier',
    'XLNetClassifier',
    'RoBERTaClassifier',
    'ALBERTClassifier',
    'ELECTRAClassifier',
    'TinyBERTClassifier',
    'FastBERTClassifier',
    'BERTBinaryClassifier',
    'XLNetBinaryClassifier',
    'RoBERTaBinaryClassifier',
    'ALBERTBinaryClassifier',
    'ELECTRABinaryClassifier',
    'BERTSeqClassifier',
    'RoBERTaSeqClassifier',
    'ALBERTSeqClassifier',
    'ELECTRASeqClassifier',
    'BERTNER',
    'BERTCRFNER',
    'BERTCRFCascadeNER',
    'BERTMRC',
    'BERTVerifierMRC',
    'RoBERTaMRC',
    'ALBERTMRC',
    'ELECTRAMRC',
    'RetroReaderMRC',
    'SANetMRC',
    'TransformerMT',

    # trial
    'BERTEMDClassifier',
    'PerformerClassifier',
]
