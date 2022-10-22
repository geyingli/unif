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
        self.transition_matrix = model.transition_matrix
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors
