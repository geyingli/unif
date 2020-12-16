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

__version__ = '2.4.2'
__date__ = '12/16/2020'


from .application import *

from .utils import load
from .utils import get_checkpoint_path
from .utils import get_assignment_map
from .utils import list_variables
from .utils import set_verbosity
from .utils import set_log


__all__ = [

    # handy methods
    'load',
    'get_checkpoint_path',
    'get_assignment_map',
    'list_variables',
    'set_verbosity',
    'set_log',
]
