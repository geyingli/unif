import os
import sys
import time

from ..third import tf
from ._base_ import Task


class Exportation(Task):
    """ Export model into PB file. """

    def run(self, export_dir, rename_inputs=None, rename_outputs=None, ignore_inputs=None, ignore_outputs=None):

        # build graph
        self.module._set_placeholders()
        _, self.module.tensors = self.module._parallel_forward(is_training=False)

        # init session
        if not self.module._session_built:
            self._init_session()
        self.module._session_mode = "infer"

        def set_input(key, value):
            inputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info("Register Input: %s, %s, %s" % (key, value.shape.as_list(), value.dtype.name))

        def set_output(key, value):
            outputs[key] = tf.saved_model.utils.build_tensor_info(value)
            tf.logging.info("Register Output: %s, %s, %s" % (key, value.shape.as_list(), value.dtype.name))

        # define inputs
        inputs = {}
        if not ignore_inputs:
            ignore_inputs = []
        for key, value in list(self.module.placeholders.items()):
            if key in ignore_inputs:
                continue
            if rename_inputs and key in rename_inputs:
                key = rename_inputs[key]
            set_input(key, value)

        # define outputs
        outputs = {}
        if not ignore_outputs:
            ignore_outputs = []
        for key, value in self.module.tensors.items():
            if key in ignore_outputs:
                continue
            if rename_outputs and key in rename_outputs:
                key = rename_outputs[key]
            set_output(key, value)

        # build signature
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs, outputs, tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        signature_def_map = {"predict": signature}
        tf.logging.info("Register Signature: predict")

        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder_path = os.path.join(export_dir, time.strftime("%Y%m%d%H%M%S"))

        # solve the path problem
        if sys.platform.startswith("win"):
            builder_path = builder_path.replace("/", "\\")

        # exportation
        try:
            builder = tf.saved_model.builder.SavedModelBuilder(builder_path)
            builder.add_meta_graph_and_variables(
                self.module.sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                legacy_init_op=legacy_init_op,
            )
        except ValueError:
            raise ValueError(
                "Twice exportation is not allowed. Try `.save()` and "
                "`.reset()` method to save and reset the graph before "
                "next exportation."
            )
        builder.save()
