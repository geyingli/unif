# coding:=utf-8
# Copyright 2021 Tencent. All rights reserved.
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

__version__ = '2.7.1'
__date__ = '2/8/2020'


from .application import *

from .utils import MultiProcess
from .utils import load
from .utils import download
from .utils import download_all
from .utils import get_checkpoint_path
from .utils import get_assignment_map
from .utils import list_variables
from .utils import list_resources
from .utils import set_verbosity
from .utils import set_log


set_verbosity()


__all__ = [

    # handy methods
    'MultiProcess',
    'load',
    'download',
    'download_all',
    'get_checkpoint_path',
    'get_assignment_map',
    'list_variables',
    'list_resources',
    'set_verbosity',
    'set_log',
]
