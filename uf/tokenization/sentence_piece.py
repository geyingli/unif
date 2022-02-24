""" SentencePiece tokenizer class.
  Code revised from XLNet team"s implementation of XLNet.
  See `https://github.com/zihangdai/xlnet`.
"""

import os
import unicodedata
from sentencepiece import SentencePieceProcessor


class SentencePieceTokenizer:
    def __init__(self, spm_file, do_lower_case=True):
        if not os.path.exists(spm_file):
            raise ValueError(
                "Can\"t find spm_file \"%s\". "
                "Please pass the correct path of sentence-piece model file, "
                "e.g.`spiece.model`." % spm_file)
        self.processor = SentencePieceProcessor()
        self.processor.Load(spm_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        text = preprocess_text(text, lower=self.do_lower_case)
        pieces = encode_pieces(self.processor, text, sample=False)
        return pieces

    def convert_tokens_to_ids(self, tokens):
        return [self.processor.PieceToId(piece) for piece in tokens]

    def convert_ids_to_tokens(self, ids):
        pieces = [self.processor.IdToPiece(_id) for _id in ids]
        return pieces


def preprocess_text(inputs, lower=False, remove_space=True,
                    keep_accents=False):
    if remove_space:
        outputs = " ".join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", """).replace("""", """)

    if not keep_accents:
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, sample=False):

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(piece[:-1].replace("▁", ""))
            if piece[0] != "▁" and cur_pieces[0][0] == "▁":
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)
    return new_pieces
