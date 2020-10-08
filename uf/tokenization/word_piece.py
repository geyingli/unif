# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' WordPiece tokenizer class.
  Code revised from Google's implementation of BERT.
  See `https://github.com/google-research/bert`.
'''

import collections
import unicodedata

from uf.tools import tf



class WordPieceTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                tokens.append(sub_token)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = convert_by_vocab(self.vocab, tokens)
        return [_id if _id else self.vocab['[UNK]'] for _id in ids]

    def convert_ids_to_tokens(self, ids):
        tokens = convert_by_vocab(self.inv_vocab, ids)
        return [_token if _token else '[UNK]' for _token in tokens]

    def add(self, char):
        index = len(self.vocab)
        self.vocab[char] = index
        self.inv_vocab[index] = char



class BasicTokenizer:
    '''Runs basic tokenization (punctuation splitting, lower casing, etc.).'''

    def __init__(self, do_lower_case=True):
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        '''Tokenizes a piece of text.'''
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        if self.do_lower_case:
            output_tokens = []
            for token in orig_tokens:
                if token.startswith('##'):
                    output_tokens.append(token)
                    continue
                token = token.lower()
                token = self._run_strip_accents(token)
                output_tokens.extend(self._run_split_on_punc(token))
            return whitespace_tokenize(' '.join(output_tokens))
        return orig_tokens

    @staticmethod
    def _run_strip_accents(text):
        '''Strips accents from a piece of text.'''
        text = unicodedata.normalize('NFD', text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == 'Mn':
                continue
            output.append(char)
        return ''.join(output)

    @staticmethod
    def _run_split_on_punc(text):
        '''Splits punctuation on a piece of text.'''
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        '''Adds whitespace around any CJK character.'''
        output = []
        for char in text:
            ord_id = ord(char)
            if _is_chinese_char(ord_id):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    @staticmethod
    def _clean_text(text):
        '''Performs invalid character removal and whitespace
        cleanup on text.'''
        output = []
        for char in text:
            ord_id = ord(char)
            if ord_id == 0 or ord_id == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)


class WordpieceTokenizer:
    '''Runs WordPiece tokenziation.'''

    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        '''Tokenizes a piece of text into its word pieces.'''

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = '##' + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def convert_to_unicode(text):
    '''Converts `text` to Unicode (if it's not already), assuming
    utf-8 input.'''
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    raise ValueError('Unsupported string type: %s' % (type(text)))


def printable_text(text):
    '''Returns text encoded in a way suitable for print or `tf.logging`.'''

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    raise ValueError('Unsupported string type: %s' % (type(text)))


def load_vocab(vocab_file):
    '''Loads a vocabulary file into a dictionary.'''
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, 'r') as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    '''Converts a sequence of [tokens|ids] using the vocab.'''
    output = []
    for item in items:
        output.append(vocab.get(item))
    return output


def whitespace_tokenize(text):
    '''Runs basic whitespace cleaning and splitting on a piece of text.'''
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    '''Checks whether `chars` is a whitespace character.'''

    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (' ', '\t', '\n', '\r'):
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False


def _is_control(char):
    '''Checks whether `chars` is a control character.'''

    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ('\t', '\n', '\r'):
        return False
    cat = unicodedata.category(char)
    if cat in ('Cc', 'Cf'):
        return True
    return False


def _is_punctuation(char):
    '''Checks whether `chars` is a punctuation character.'''
    ord_id = ord(char)

    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as '^', '$', and '`' are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (ord_id >= 33 and ord_id <= 47) or \
            (ord_id >= 58 and ord_id <= 64) or \
            (ord_id >= 91 and ord_id <= 96) or \
            (ord_id >= 123 and ord_id <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith('P'):
        return True
    return False


def _is_chinese_char(ord_id):
    '''Checks whether ord_id is the codepoint of a CJK character.'''
    # This defines a `Chinese character` as anything in the CJK
    # Unicode block:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and
    # Korean characters, despite its name. The modern Korean Hangul
    # alphabet is a different block, as is Japanese Hiragana and
    # Katakana. Those alphabets are used to write space-separated
    # words, so they are not treated specially and handled like the
    # all of the other languages.
    if (ord_id >= 0x4E00 and ord_id <= 0x9FFF) or \
            (ord_id >= 0x3400 and ord_id <= 0x4DBF) or \
            (ord_id >= 0x20000 and ord_id <= 0x2A6DF) or \
            (ord_id >= 0x2A700 and ord_id <= 0x2B73F) or \
            (ord_id >= 0x2B740 and ord_id <= 0x2B81F) or \
            (ord_id >= 0x2B820 and ord_id <= 0x2CEAF) or \
            (ord_id >= 0xF900 and ord_id <= 0xFAFF) or \
            (ord_id >= 0x2F800 and ord_id <= 0x2FA1F):
        return True
    return False