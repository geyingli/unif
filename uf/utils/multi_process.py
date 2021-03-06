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

import multiprocessing

from ..tools import tf

NUM_PROCESSES = 1
pool = None


class MultiProcess:
    def __init__(self, n_process='auto'):
        n_cpu = multiprocessing.cpu_count()
        if n_process != 'auto':
            assert n_process <= n_cpu, (
                'Invalid value of `n_process`. It can not exceed the num '
                'of cpu cores in the device: %d.' % n_cpu)
        else:
            n_process = n_cpu
        self.n = n_process

    def __enter__(self):
        global NUM_PROCESSES, pool
        if self.n > 1:
            pool = multiprocessing.Pool(self.n)
        NUM_PROCESSES = self.n

    def __exit__(self, *args, **kwargs):
        global NUM_PROCESSES, pool
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        NUM_PROCESSES = 1


def _parallel_convert_single_process(args):
    bucket_id = args[0]
    app_class = args[1]
    mapping = args[2]
    data = args[3]
    is_training = args[4]

    # Verbosity of tensorflow in new process will be set to default,
    # for this reason we just have to silence the logging and don't
    # have to care about the recovery.
    tf.logging.set_verbosity(tf.logging.FATAL)
    model = app_class(*mapping)

    data = model.convert(
        data['X'], data['y'], data['sample_weight'], data['X_tokenized'],
        is_training, True)
    return (bucket_id, data)
