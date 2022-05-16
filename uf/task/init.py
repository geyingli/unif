import os

from ..third import tf
from .. import com
from .base import Task


class Initialization(Task):

    def __init__(self, module):
        self.module = module

        self.decorate()

    def decorate(self):
        self.module._set_placeholders("placeholder", is_training=False)

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self, reinit_all, ignore_checkpoint):

        # init session
        if not self.module._session_built:
            com.count_params(self.module.global_variables, self.module.trainable_variables)
            self._init_session(ignore_checkpoint)
        elif reinit_all:
            self._init_session(ignore_checkpoint)
        else:
            variables = []
            for var in self.module.global_variables:
                if var not in self.module._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables, ignore_checkpoint)
            else:
                tf.logging.info("Global variables already initialized. To re-initialize all, pass `reinit_all` to True.")
        self.module._session_mode = "infer"

    def _init_session(self, ignore_checkpoint):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.module._gpu_ids)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.module.sess = tf.Session(graph=self.module.graph, config=config)
        self._init_variables(self.module.global_variables, ignore_checkpoint)
        self.module._session_built = True
