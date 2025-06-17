"""
Constants and general methods.

Constants:
    * WHITE: pattern matching whitespaces.
    * SYMBOLS_GROUP: pattern group matching symbols.
    * NUMBER_GROUP: pattern group matching numbers.
    * WHITE_CHAR: ▁
    * SPECIAL_DICT: dictionary with unicode codepoints to remove ({codepoint: None}).
"""

from __future__ import annotations
import os
import sys
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

TokenMask = tuple[int]
# INF = np.inf
INF = int(10e9)
WHITE = re.compile(r"\s+")  # Match whitespaces
WHITE_CHAR = "▁"
SYMBOLS_GROUP = re.compile(rf"({WHITE_CHAR}+[^\w\s]|[^\w\s{WHITE_CHAR}])") #white+punctuation or punctuation
NUMBER_GROUP = re.compile(r"([0-9]+)")


_CATEGORIES = ["Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"]
SPECIAL_DICT: dict[int, None] = dict.fromkeys(  # ~40 MB
    [i for i in range(sys.maxunicode) if unicodedata.category(chr(i)) in _CATEGORIES]
)


def prepare(line: str) -> str:
    """
    Args:
        line: sentence.

    Returns:
        Clean sentence.
    """
    line = unicodedata.normalize("NFKC", line)  # Unicode Normalize
    line = line.translate(SPECIAL_DICT)
    line = re.sub(NUMBER_GROUP, r" \1 ", line)  # Space around number sequences
    line = re.sub(SYMBOLS_GROUP, r" \1 ", line)  # Space around punctuation
    line = re.sub(WHITE, " ", line)  # Space normalization
    line = line.strip()  # Strip
    return line


def prepare_src(line: str) -> list[str]:
    """
    Args:
        line: tokenized sentence in the source language.

    Returns:
        List of tokens.
    """
    line = prepare(line)
    return line.split()


def prepare_trg(line: str) -> str:
    """
    Args:
        line: sentence in the target language.

    Returns:
        Clean sentence.
    """
    line = prepare(line)
    return re.sub(WHITE, WHITE_CHAR, f"{WHITE_CHAR}{line}")  # ▁

def prepare_align(line: str) -> list[tuple[int, int]]:
    pairs = line.split()
    pairs = [i.split("-") for i in pairs]
    return [(int(i[0]), int(i[1])) for i in pairs]

def iter_spans(
    line: str,
    maxlen: int = INF,
    minlen: int = 1,
    pretokenize: bool = False,
    white_char: str = WHITE_CHAR,
) -> Iterator[str]:
    """
    List all the substring in the sequence.

    Args:
        line: string in the target language.
        maxlen: maximum length of a span.
        minlen: minimum length of a span.
        pretokenize: spans do not cross word boundaries (▁).

    Returns:
        List of spans
    """
    length = len(line)
    maxlen = min(maxlen, length)
    for i in range(length + 1):  # Start position
        for l in range(minlen, min(length - i, maxlen) + 1):
            # Cross word boundary, whitespace at the beginning ok
            if pretokenize and white_char in line[i + 1 : i + l]:
                # Extending the span is useless
                break
            yield line[i : i + l]


def iter_spans_index(
    line: str,
    maxlen: int = INF,
    minlen: int = 1,
    pretokenize: bool = False,
    white_char: str = WHITE_CHAR,
) -> Iterator[str]:
    """
    List all the substring in the sequence.

    Args:
        line: string in the target language.
        maxlen: maximum length of a span.
        minlen: minimum length of a span.
        pretokenize: spans do not cross word boundaries (▁).

    Returns:
        List of spans
    """
    length = len(line)
    maxlen = min(maxlen, length)
    for i in range(length + 1):  # Start position
        for l in range(minlen, min(length - i, maxlen) + 1):
            # Cross word boundary, whitespace at the beginning ok
            if pretokenize and white_char in line[i + 1 : i + l]:
                # Extending the span is useless
                break
            yield i, i + l, line[i : i + l]


def list_spans(
    line: str,
    maxlen: int = INF,
    minlen: int = 1,
    pretokenize: bool = False,
) -> list[str]:
    """
    List all the substring in the sequence.

    Args:
        line: string in the target language.
        maxlen: maximum length of a span.
        minlen: minimum length of a span.
        pretokenize: spans do not cross word boundaries (▁).

    Returns:
        List of spans
    """
    return list(iter_spans(line, maxlen, minlen, pretokenize))


def set_spans(
    line: str,
    maxlen: int = INF,
    minlen: int = 1,
    pretokenize: bool = False,
) -> set[str]:
    """
    Args:
        line: string in the target language.
        maxlen: maximum length of a span.
        minlen: minimum length of a span.
        pretokenize: spans do not cross word boundaries (▁).

    Returns:
        Set of spans
    """
    return set(iter_spans(line, maxlen, minlen, pretokenize))


def to_file(file: str | None, gen: Iterable[Any], fmt: Callable[[Any], str]) -> None:
    """
    Write to file.

    Args:
        file: file path. STDOUT if None.
        gen: iterable with the items to print.
        fmt: function to format the items as string.
    """
    if file is None:
        for i in gen:
            print(fmt(i))
    else:
        dir_ = os.path.dirname(file)
        if dir_ != "":
            os.makedirs(dir_, exist_ok=True)
        with open(file, "w") as f:
            for i in gen:
                f.write(f"{fmt(i)}\n")


def banded_matrix(x1: int, x2: int, dtype: npt.DTypeLike = np.uint32) -> npt.NDArray:
    """
    Args:
        x1: first dimension.
        x2: second dimension.
        dtype: numpy dtype.

    Returns:
        Banded rectangular matrix. The band includes the two opposite corners.
    """
    diff = x2 - x1  # Neg. band size
    # Build lower triangular mat., then take upper triangular
    return np.triu(np.tri(x1, x2, max(diff, 0), dtype=dtype), min(diff, 0))


def band_count(
    spans: list[int], tokens: list[int], dtype: npt.DTypeLike
) -> npt.NDArray[np.uint]:
    """
    Count the cooccurency of the spans and tokens in the banded region.

    Args:
        spans: list of spans in the target sentence.
        tokens: list of tokens in the source sentence.
        dtype: numpy dtype.

    Returns
        Count of the unique span-tokens sorted by id.
    """
    band = banded_matrix(len(spans), len(tokens), dtype)
    # Inverse: index in the unique array
    u_spans, inv_spans = np.unique(spans, return_inverse=True)
    span_range = np.arange(len(u_spans))
    u_tokens, inv_tokens = np.unique(tokens, return_inverse=True)
    token_range = np.arange(len(u_tokens))

    # spans x unique spans
    # i-th column: row in banded mat refers to the i-th span
    # Binary matrix
    span_groups = np.repeat(inv_spans.reshape(-1, 1), len(span_range), 1) == span_range
    span_groups = span_groups.astype(dtype)

    # tokens x unique tokens
    token_groups = (
        np.repeat(inv_tokens.reshape(-1, 1), len(token_range), 1) == token_range
    )
    token_groups = token_groups.astype(dtype)

    # Somehow it works
    # unique spans x unique tokens
    return span_groups.T @ band @ token_groups


def invert_dict(d: dict[Any, Any]) -> dict[Any, Any]:
    return {v: k for k, v in d.items()}

def empty_min(x: Iterable[Any | None]) -> Any | None:
    min_ = np.inf
    empty = True
    all_none = True
    for i in x:
        empty = False
        if i is not None and i < min_:
            all_none = False
            min_ = i
    if all_none or empty:
        return None
    return min_

def empty_max(x: Iterable[Any | None]) -> Any | None:
    max_ = -np.inf
    empty = True
    all_none = True
    for i in x:
        empty = False
        if i is not None and i > max_:
            all_none = False
            max_ = i
    if all_none or empty:
        return None
    return max_
