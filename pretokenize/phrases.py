# Adapted from nltk.translate.phrase_based
#
#  Natural Language Toolkit: Phrase Extraction Algorithm
#
# Copyright (C) 2001-2024 NLTK Project
# Authors: Liling Tan, Fredrik Hedman, Petra Barancikova
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from collections.abc import Iterator
import logging

logger = logging.getLogger(__name__)


def _phrase_extraction(
    src: list[str], trg: list[str], alignment: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """
    Phrase extraction algorithm extracts the longest contiguous phrase pairs from
    a word-aligned sentence pair.

    The idea is to find a pair of phrases and try to extend the source phrase so that it is still aligned.

    Args:
        src: source tokens.
        trg: target tokens.
        alignment: alignment as list of index pairs.

    Returns:
        List of phrases pairs as tuple of indexes: (src start, src end, trg start, trg end).
        The tokens can be obtained as src[src start : src end + 1].
    """
    slen = len(src)
    tlen = len(trg)
    phrases = []

    s_start = 0
    s_end = 0
    current_phrase = None
    while s_start < slen and s_end < slen:
        # If s_start-s_end has a phrase: try to extend it
        # Otherwise: add the current phrase to the table and increase s_start by its length
        t_start = tlen - 1
        t_end = -1
        valid = True
        # Find trg boundaries
        for s, t in alignment:
            if s_start <= s <= s_end:  # Alignment inside the src phrase
                t_start = min(t, t_start)
                t_end = max(t, t_end)
        # Check that no trg token is aligned to src tokens outside the phrase
        for s, t in alignment:
            if (s < s_start or s > s_end) and (t_start <= t <= t_end):
                valid = False
                break
        if valid:  # Try to extend
            current_phrase = (s_start, s_end, t_start, t_end)
        else:  # Current phrase is the longest valid
            if current_phrase is not None:
                phrases.append(current_phrase)
            current_phrase = None
            s_start = s_end
        s_end += 1  # Extend
    if s_end >= slen:  # Reached the end
        if current_phrase is not None:
            phrases.append(current_phrase)
    return phrases


def extract_phrases(
    src: list[str], trg: list[str], alignment: list[str]
) -> Iterator[tuple[str, str]]:
    """
    Args:
        src: list of source sentences.
        trg: list of target sentences.
        alignments: list of eflomal alignmes (e.g, 0-0 1-2 3-3 ..., zero indexed).

    Returns:
        Iterator with pair of parallel phrases.
    """
    for s, t, a in zip(src, trg, alignment):
        pairs = a.split()
        pairs = [(int(i[0]), int(i[1])) for i in map(lambda x: x.split("-"), pairs)]
        s_tok = s.split()
        t_tok = t.split()
        logger.debug("Extract phrases")
        phrases = _phrase_extraction(s_tok, t_tok, pairs)
        logger.debug("Write phrases")
        for s_start, s_end, t_start, t_end in phrases:
            phrase_s = s_tok[s_start : s_end + 1]
            phrase_t = t_tok[t_start : t_end + 1]
            yield " ".join(phrase_s), " ".join(phrase_t)


if __name__ == "__main__":
    """
    ((0, 4), (0, 7), 'michael assumes that he', 'michael geht davon aus , dass er')
    ((4, 6), (9, 10), 'will stay', 'bleibt')
    # ((4, 9), (7, 10), 'will stay in the house', 'im haus bleibt')
    ((6, 9), (7, 9), 'in the house', 'im haus')
    """
    alignment = [
        (0, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 5),
        (3, 6),
        (4, 9),
        (5, 9),
        (6, 7),
        (7, 7),
        (8, 8),
    ]
    trg = "michael geht davon aus , dass er im haus bleibt"
    src = "michael assumes that he will stay in the house"
    # alignment = [(3,0), (2,1), (1,2), (0,3)]
    # trg = "0 1 2 3"
    # src = "0 1 2 3"
    phrases = sorted(_phrase_extraction(src.split(), trg.split(), alignment))
    for i in phrases:
        print(i)
        s_start, s_end, t_start, t_end = i
        phrase_s = src.split()[s_start : s_end + 1]
        phrase_t = trg.split()[t_start : t_end + 1]
        print(" ".join(phrase_s))
        print(" ".join(phrase_t))

