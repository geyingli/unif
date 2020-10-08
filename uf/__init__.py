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

__version__ = '2.1.25'
__date__ = '10/5/2020'


from .application.bert import BERTLM
from .application.roberta import RoBERTaLM
from .application.albert import ALBERTLM
from .application.electra import ELECTRALM
from .application.gpt2 import GPT2LM
from .application.dilated import DilatedLM
from .application.bert import BERTClassifier
from .application.xlnet import XLNetClassifier
from .application.roberta import RoBERTaClassifier
from .application.albert import ALBERTClassifier
from .application.electra import ELECTRAClassifier
from .application.tiny_bert import TinyBERTClassifier
from .application.fast_bert import FastBERTClassifier
from .application.bert import BERTBinaryClassifier
from .application.xlnet import XLNetBinaryClassifier
from .application.roberta import RoBERTaBinaryClassifier
from .application.albert import ALBERTBinaryClassifier
from .application.electra import ELECTRABinaryClassifier
from .application.bert import BERTSeqClassifier
from .application.roberta import RoBERTaSeqClassifier
from .application.albert import ALBERTSeqClassifier
from .application.electra import ELECTRASeqClassifier
from .application.bert import BERTNER
from .application.bert import BERTCRFNER
from .application.bert import BERTCRFCascadeNER
from .application.bert import BERTMRC
from .application.roberta import RoBERTaMRC
from .application.albert import ALBERTMRC
from .application.electra import ELECTRAMRC
from .application.transformer import TransformerMT

from .utils import get_checkpoint_path
from .utils import get_assignment_map
from .utils import set_log

from . import modeling
from . import tokenization


__all__ = [

    # application classes
    'BERTLM',
    'RoBERTaLM',
    'ALBERTLM',
    'ELECTRALM',
    'GPT2LM',
    'DilatedLM',
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
    'RoBERTaMRC',
    'ALBERTMRC',
    'ELECTRAMRC',
    'TransformerMT',

    # useful methods
    'get_checkpoint_path',
    'get_assignment_map',
    'set_log',

    # self-graph building
    'modeling',
    'tokenization',
]
