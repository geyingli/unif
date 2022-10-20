import collections


class BaseEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def get_pooled_output(self):
        raise NotImplementedError()

    def get_sequence_output(self):
        raise NotImplementedError()


class BaseDecoder:
    def __init__(self, *args, **kwargs):

        # scalar of total loss, used for back propagation
        self.train_loss = None

        # supervised tensors of each example
        self._tensors = collections.OrderedDict()

    def get_forward_outputs(self):
        return (self.train_loss, self._tensors)