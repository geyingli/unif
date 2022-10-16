import numpy as np

from ...core import BaseModule
from ... import com


class MTModule(BaseModule):
    """ Application class of machine translation (MT). """

    def _get_bleu(self, preds, labels, mask, max_gram=4):
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

            _rouge = nominator / (denominator + 1e-6)
            rouges.append(_rouge)

        return np.mean(rouges)
