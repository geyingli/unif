from ...core import BaseModule


class RegressorModule(BaseModule):
    """ Application class of regression. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "max_seq_length": "An integer that defines max sequence length of input tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }
