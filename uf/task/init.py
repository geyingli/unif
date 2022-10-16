import os

from ..third import tf
from .. import com
from .base import Task


class Initialization(Task):
    """ Initialze the model, make it ready for inference. """

    def __init__(self, module):
        self.module = module

        self.decorate()

    def decorate(self):
        self.module._set_placeholders()

        _, self.module._tensors = self.module._parallel_forward(False)

    def run(self, reinit_all, ignore_checkpoint):

        # init session
        if reinit_all:
            self._init_session(ignore_checkpoint=ignore_checkpoint)
        
        # init uninitialized variables
        else:
            variables = []
            for var in self.module.global_variables:
                if var not in self.module._inited_vars:
                    variables.append(var)
            if variables:
                self._init_variables(variables, ignore_checkpoint=ignore_checkpoint)
            else:
                tf.logging.info(
                    "Global variables already initialized. To re-initialize all, "
                    "pass `reinit_all` to True."
                )
        self.module._session_mode = "infer"
