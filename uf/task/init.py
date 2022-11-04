from ..third import tf
from ._base_ import Task


class Initialization(Task):
    """ Initialze the model, make it ready for inference. """

    def run(self, reinit_all, ignore_checkpoint):

        # build graph
        if self.module._graph_mode is None:
            self._build_graph()

        # init session
        if reinit_all or not self.module._session_built:
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
