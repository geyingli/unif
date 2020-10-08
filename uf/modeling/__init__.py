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


from .bert import BERTEncoder, BERTDecoder
from .xlnet import XLNetEncoder
from .albert import ALBERTEncoder, ALBERTDecoder
from .electra import ELECTRA
from .gpt2 import GPT2
from .dilated import DLM
from .tiny_bert import TinyBERTCLSDistillor
from .fast_bert import FastBERTCLSDistillor
from .base import CLSDecoder, BinaryCLSDecoder, SeqCLSDecoder, MRCDecoder
from .crf import CRFDecoder
from .transformer import Transformer


__all__ = [
    'BERTEncoder',
    'BERTDecoder',
    'XLNetEncoder',
    'ALBERTEncoder',
    'ALBERTDecoder',
    'ELECTRA',
    'GPT2',
    'DLM',
    'TinyBERTCLSDistillor',
    'FastBERTCLSDistillor',
    'CLSDecoder',
    'BinaryCLSDecoder',
    'SeqCLSDecoder',
    'MRCDecoder',
    'CRFDecoder',
    'Transformer',
]