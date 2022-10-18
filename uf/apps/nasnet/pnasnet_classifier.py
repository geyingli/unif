import numpy as np

from .pnasnet import build_pnasnet_mobile, build_pnasnet_large
from .._base_._base_classifier import ClassifierModule
from .._base_._base_ import ClsDecoder
from ..bert.bert_classifier import BERTClassifier
from ...third import tf


class PNasNetClassifier(BERTClassifier, ClassifierModule):
    """ Single-label classifier on PNasNet. """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "label_size": "An integer that defines number of possible labels of outputs",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def __init__(
        self,
        image_size=128,
        label_size=None,
        init_checkpoint=None,
        output_dir=None,
        gpu_ids=None,
        model_size="large",
        data_format="NHWC",
    ):
        self.__init_args__ = locals()
        super(ClassifierModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.image_size = image_size
        self.label_size = label_size
        self.model_size = model_size
        self.data_format = data_format
        self._id_to_label = []

        assert model_size in ("mobile", "large"), (
            f"Invalid `model_size`: {model_size}. Pick one from \"mobile\" and \"large\".")
        self.decay_power = "unsupported"

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        assert not X_tokenized, "%s does not support text input." % self.__class__.__name__ 
        if is_training:
            assert y is not None, "`y` can't be None."
        if is_parallel:
            assert self.label_size, "Can't parse data on multi-processing when `label_size` is None."

        n_inputs = None
        data = {}

        # convert X
        if X:
            input_ids = self._convert_X(X)
            data["input_ids"] = np.array(input_ids, dtype=np.int32)
            n_inputs = len(input_ids)

            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            label_ids = self._convert_y(y)
            data["label_ids"] = np.array(label_ids, dtype=np.int32)

        # convert sample_weight
        if is_training or y:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X):

        # convert to numpy array
        image_arrays = []
        for idx, sample in enumerate(X):
            try:
                image_arrays.append(self._convert_x(sample))
            except:
                raise ValueError("Wrong input format (image %d): \"%s\". " % (idx, sample))
        
        # interpolate

        return np.vstack(image_arrays)

    def _convert_x(self, x):
        return np.array(x)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input_ids": tf.placeholder(tf.float32, [None, 3, self.image_size, self.image_size], "input_ids"),
            "label_ids": tf.placeholder(tf.int32, [None], "label_ids"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):
        
        if self.model_size == "mobile":
            logits, _ = build_pnasnet_mobile(
                images=placeholders["input_ids"], num_classes=self.label_size,
                is_training=is_training, final_endpoint=None,
            )
        elif self.model_size == "large":
            logits, _ = build_pnasnet_large(
                images=placeholders["input_ids"], num_classes=self.label_size,
                is_training=is_training, final_endpoint=None,
            )

        decoder = ClsDecoder(
            is_training,
            logits,
            placeholders["label_ids"],
            is_logits=True,
            label_size=self.label_size,
            sample_weight=placeholders.get("sample_weight"),
        )   
        return decoder.get_forward_outputs()
