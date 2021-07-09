# coding:=utf-8
# Copyright 2021 Tencent. All rights reserved.
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

import os
import collections
import unicodedata

from ..utils import is_whitespace, is_control, is_punctuation, is_chinese_char
from ..tools import tf


def get_word_piece_tokenizer(vocab_file, do_lower_case=True):
    if not os.path.exists(vocab_file):
        raise ValueError(
            'Can\'t find vocab_file \'%s\'. '
            'Please pass the correct path of vocabulary file, '
            'e.g.`vocab.txt`.' % vocab_file)
    return WordPieceTokenizer(vocab_file, do_lower_case=do_lower_case)


class WordPieceTokenizer:
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)  # word: idx
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.processor = [BasicTokenizer(do_lower_case=do_lower_case),
                          WordpieceTokenizer(vocab=self.vocab)]

    def tokenize(self, text):
        tokens = []
        for token in self.processor[0].tokenize(text):
            for sub_token in self.processor[1].tokenize(token):
                tokens.append(sub_token)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = convert_by_vocab(self.vocab, tokens)
        return [_id if _id else self.vocab.get('[UNK]', 0) for _id in ids]

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
            if is_punctuation(char):
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
            if is_chinese_char(ord_id):
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
            if ord_id == 0 or ord_id == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
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
        '''Tokenizes a piece of text into its word pieces.
        NOTE(geyingli): we do not create `unk_token` in this step.'''

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # output_tokens.append(self.unk_token)
                output_tokens.append(token)
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
                # output_tokens.append(self.unk_token)
                output_tokens.append(token)
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
