import numpy as np

from .chatbot import Chatbot
from ..transformer.transformer_mt import TransformerMT
from .._base_._base_mt import MTModule


class ChatbotMT(TransformerMT, MTModule):
    """ Chatbot. """

    def _forward(self, is_training, placeholders, **kwargs):

        model = Chatbot(
            vocab_size=len(self.tokenizer.vocab),
            is_training=is_training,
            source_ids=placeholders["source_ids"],
            target_ids=placeholders["target_ids"],
            sos_id=self.tokenizer.convert_tokens_to_ids(["<s>"])[0],
            sample_weight=placeholders.get("sample_weight"),
            hidden_size=self._hidden_size,
            num_blocks=self._num_hidden_layers,
            num_attention_heads=self._num_attention_heads,
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["losses"]]
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        # loss
        batch_losses = output_arrays[0]
        loss = np.mean(batch_losses)

        info = ""
        info += ", loss %.6f" % loss

        return info
