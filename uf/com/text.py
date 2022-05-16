import re
import unicodedata


def convert_tokens_to_text(tokens):
    words = [""]
    for _token in tokens:
        if _token.startswith("##"):
            words[-1] += _token[2:]
        else:
            words.append(_token)
    text = " ".join(words)

    # remove spaces
    if len(text) >= 3:
        i = 1
        while i < len(text) - 1:
            if is_whitespace(text[i]):
                _last = text[i - 1]
                _next = text[i + 1]

                # remove space between chars and punctuations
                if not is_english_char(_last) or not is_english_char(_next):
                    text = text.replace("%s%s%s" % (_last, text[i], _next), "%s%s" % (_last, _next))
            i += 1

    return text.strip()


def align_tokens_with_text(tokens, text, lower_case):
    if lower_case:
        text = text.lower()

    i = 0
    j = 0
    max_j = len(text)
    mapping_start = []
    mapping_end = []
    while i < len(tokens):
        token = tokens[i]
        token = token.replace("##", "")
        if text[j:].startswith(token):
            mapping_start.append(j)
            mapping_end.append(j + len(token))
            i += 1
            j += len(token)
        elif token not in text[j:]:  # [CLS], [SEP], some Japanese signs
            mapping_start.append(j)
            if token in ("[CLS]", "[SEP]"):
                mapping_end.append(j)
            else:
                mapping_end.append(j + len(token))
            i += 1
        else:
            j += 1
        if j >= max_j:
            break

    for _ in range(len(tokens) - len(mapping_start)):
        mapping_start.append(max_j + 1000)
        mapping_end.append(max_j + 1000)

    return mapping_start, mapping_end


def find_boyer_moore(T, P, start=0):
    """ BM algorithm for string match. """

    n, m = len(T), len(P)
    last = {}
    for k in range(m):
        last[P[k]] = k

    # align end of pattern at index m-1 of text
    i = start + m - 1
    k = m - 1
    while i < n:
        if T[i] == P[k]:
            if k == 0:
                return i
            i -= 1
            k -= 1
        else:
            j = last.get(T[i], -1)
            i += m - min(k, j + 1)
            k = m - 1
    return -1


def find_all_boyer_moore(T, P):
    start_ids = []
    start = 0
    while True:
        start_position = find_boyer_moore(
            T, P, start=start)
        if start_position == -1:
            break
        start_ids.append(start_position)
        start = start_position + len(P)
    return start_ids


def is_english_char(char):
    if re.findall("[a-zA-Z]", char):
        return True
    return False


def is_numeric_char(char):
    if re.findall(r"[\d]", char):
        return True
    return False


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""

    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in (" ", "\t", "\n", "\r"):
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""

    # These are technically control characters but we count them as whitespace
    # characters.
    if char in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    ord_id = ord(char)

    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (ord_id >= 33 and ord_id <= 47) or \
            (ord_id >= 58 and ord_id <= 64) or \
            (ord_id >= 91 and ord_id <= 96) or \
            (ord_id >= 123 and ord_id <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def is_chinese_char(ord_id):
    """Checks whether ord_id is the codepoint of a CJK character."""
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
