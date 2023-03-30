import copy
import numpy as np

from ...core import BaseModule
from ... import com


class MTModule(BaseModule):
    """ Application class of machine translation (MT). """

    _INFER_ATTRIBUTES = {    # params whose value cannot be None in order to infer without training
        "source_max_seq_length": "An integer that defines max sequence length of source language tokens",
        "target_max_seq_length": "An integer that defines max sequence length of target language tokens",
        "init_checkpoint": "A string that directs to the checkpoint file used for initialization",
    }

    def _get_bleu(self, preds, labels, mask, max_gram=4):
        """ Bilingual evaluation understudy. """
        eos_id = self.tokenizer.convert_tokens_to_ids(["</s>"])[0]

        bleus = []
        for _preds, _labels, _mask in zip(preds, labels, mask):

            # preprocess
            for i in range(len(_preds)):
                if _preds[i] == eos_id:
                    _preds = _preds[:i+1]
                    break
            _labels = _labels[:int(np.sum(_mask)) - 1]  # remove </s>

            power = 0
            for n in range(max_gram):
                ngrams = []
                nominator = 0
                denominator = 0

                for i in range(len(_labels) - n):
                    ngram = _labels[i:i+1+n].tolist()
                    if ngram in ngrams:
                        continue
                    cand_count = len(com.find_all_boyer_moore(_preds, ngram))
                    ref_count = len(com.find_all_boyer_moore(_labels, ngram))
                    nominator += min(cand_count, ref_count)
                    denominator += cand_count
                    ngrams.append(ngram)

                power += 1 / (n + 1) * np.log(nominator / (denominator + 1e-6) + 1e-6)

            _bleu = np.exp(power)
            if len(_preds) >= len(_labels):
                _bleu *= np.exp(1 - len(_labels) / len(_preds))
            bleus.append(_bleu)

        return np.mean(bleus)

    def _get_rouge(self, preds, labels, mask, max_gram=4):
        """ Recall-Oriented Understudy for Gisting Evaluation. """
        eos_id = self.tokenizer.convert_tokens_to_ids(["</s>"])[0]

        rouges = []
        for _preds, _labels, _mask in zip(preds, labels, mask):

            # preprocess
            for i in range(len(_preds)):
                if _preds[i] == eos_id:
                    _preds = _preds[:i+1]
                    break
            _labels = _labels[:int(np.sum(_mask)) - 1]  # remove </s>

            nominator = 0
            denominator = 0
            for n in range(max_gram):
                ngrams = []

                for i in range(len(_labels) - n):
                    ngram = _labels[i:i+1+n].tolist()
                    if ngram in ngrams:
                        continue
                    nominator += len(com.find_all_boyer_moore(_preds, ngram))
                    denominator += len(com.find_all_boyer_moore(_labels, ngram))
                    ngrams.append(ngram)

            _rouge = nominator / denominator if denominator else 0
            rouges.append(_rouge)

        return np.mean(rouges)

    def _convert_x(self, x, tokenized):

        # deal with untokenized inputs
        if not tokenized:

            # deal with general inputs
            if isinstance(x, str):
                return self.tokenizer.tokenize(x)

        # deal with tokenized inputs
        elif isinstance(x[0], str):
            return copy.deepcopy(x)

        # deal with tokenized and multiple inputs
        raise ValueError("Machine translation module only supports single sentence inputs.")
