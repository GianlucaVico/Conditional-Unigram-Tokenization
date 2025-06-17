from collections import Counter
from collections.abc import Callable, Iterable
import io
import functools
import tarfile
from typing import IO, Self

import ipdb
from tqdm import tqdm

from ..utils import iter_spans, TokenMask, INF
from .trie import SpanIndex, TokenIndex, CountTable, MAX_CACHE

import logging

logger = logging.getLogger(__name__)


class Table:
    def __init__(self) -> None:
        """
        Table containg the co-occourrences of spans and tokens.
        The spans and tokens can be used as strings.
        The table is made up of three components:
        - SpanIndex: maps spans to span ids
        - TokenIndex: maps tokens to token ids
        - CountTable: maps pairs of span id and token id to a count

        Attributes:
            lock_alphabet: disallow spans with characters not in the alphabet.
        """
        logger.debug("Create table")
        # Table: span index, token index, count
        self._span_index = SpanIndex()
        self._token_index = TokenIndex()
        self._count = CountTable()

        self._inv_token_index = None
        self._inv_span_index = None

        self.lock_alphabet: bool = False

    def __iadd__(self, other: Self) -> Self:
        """Add two tables WITH THE SAME INDEXES"""
        if (
            self._span_index != other._span_index
            or self._token_index != other._token_index
        ):
            raise ValueError("Indexes do not match")
        self.add(other)
        return self

    def __imul__(self, other: int) -> Self:
        self._count *= other
        self.cache_clear()
        return self

    def __itruediv__(self, other: int) -> Self:
        return self.__imul__(1 / other)

    def __ifloordiv__(self, other: int) -> Self:
        self._count //= other
        self.cache_clear()
        return self

    def get_linked_empty(self) -> Self:
        """
        Returns:
            A new table with the same indexes but empty counts.
        """
        table = self.__class__()
        table._span_index = self._span_index
        table._token_index = self._token_index
        table._inv_span_index = self._inv_span_index
        table._inv_token_index = self._inv_token_index
        return table

    def add(self, other: Self) -> None:
        """Increase the count table"""
        self._count.add(other._count)
        self.cache_clear()

    @functools.cached_property
    def alphabet(self) -> set[str]:
        return self._span_index.alphabet

    @property
    def spans(self) -> Iterable[str]:
        """
        Returns:
            List of spans.
        """
        return self._span_index.todict().keys()

    @property
    def tokens(self) -> Iterable[str]:
        """
        Returns:
            List of tokens.
        """
        return self._token_index.keys()

    def get_token_id(self, token: str) -> int:
        """Get the id of a token"""
        return self._token_index[token]

    def get_span_id(self, span: str, default: int | None = None) -> int | None:
        """Get the id of a span. Returns default of the span is not in the table."""
        return self._span_index.get(span, default)

    def get_token(self, token_id: int, default: str | None = None) -> str | None:
        """Get the token for the id. Returns default of the token is not in the table."""
        return self._inv_token_index.get(token_id, default)

    def get_span(self, span_id: int, default: str | None = None) -> str | None:
        """Get the span for the id. Returns default of the span is not in the table."""
        return self._inv_span_index.get(span_id, default)

    @functools.lru_cache(MAX_CACHE)
    def get(self, span: str, token: str, default: int = 0) -> int:
        """
        Get the count for the pair of span and token.
        Returns default if either the span or the token is not in the table.
        Notes: cached
        """
        s_id = self._span_index.get(span, None)
        t_id = self._token_index[token]
        if s_id is None or t_id is None:
            return default
        return self._count[s_id, t_id]

    @functools.lru_cache(MAX_CACHE)
    def __getitem__(self, key: tuple[str, str]) -> int:
        """
        Get the count for the pair of span and token.
        Notes: cached
        """
        s_id = self._span_index[key[0]]
        t_id = self._token_index[key[1]]
        return self._count[s_id, t_id]

    @functools.lru_cache(MAX_CACHE)
    def get_by_id(
        self, span_id: int, token_ids: TokenMask | None = None, default: int = 0
    ) -> list[int]:
        """
        Select a row in the count table.

        Args:
            span_id: id of the span. Must be in the table.
            token_ids: id of the tokens to select. If None, select all the tokens.

        Returns:
            List of counts.

        Notes: cached
        """
        values = self._count[span_id] # dict {id: count}
        if token_ids is None:
            return list(values.values())
        return [values.get(i, default) for i in token_ids]
    
    @functools.lru_cache(MAX_CACHE)
    def get_dict_by_id(self, span_id: int, token_ids: TokenMask | None = None) -> dict[int, int]:
        """
        Select a row in the count table.

        Args:
            span_id: id of the span. Must be in the table.
            token_ids: id of the tokens to select. If None, select all the tokens.

        Returns:
            Dictionary of counts.

        Notes: cached
        """
        values = self._count[span_id]
        if token_ids is None:
            return values
        return {i: values.get(i, 0) for i in token_ids}

    def get_token_mask(self, tokens: list[str]) -> TokenMask:
        """Returns the token ids. Ignore the tokens that are not in the table."""
        tmp = [self._token_index[i] for i in tokens]
        return tuple([i for i in tmp if i is not None])

    def get_span_mask(self, spans: list[str]) -> tuple[int]:
        """Returns the spans ids. Ignore the spans that are not in the table."""
        tmp = [self._span_index[i] for i in spans]
        return tuple([i for i in tmp if i is not None])

    @functools.lru_cache(MAX_CACHE)
    def get_token_count(self, token_ids: TokenMask) -> list[int]:
        """
        Marginalize over the spans and returns the counts for the tokens.

        Notes: cached.
        """
        return [self.count_tok.get(i, 0) for i in token_ids]

    def __setitem__(self, key: tuple[str, str], value: int) -> None:
        """
        Insert or update a pair in the table.
        """
        if self.lock_alphabet and set.issuperset(self.alphabet, key[0]):
            return  # Ignore invalid span
        span_index = self._span_index[key[0]]
        token_index = self._token_index[key[1]]
        self._count[span_index, token_index] = value
        self.cache_clear()

    def __delitem__(self, key: tuple[str, str]) -> None:
        """Delete a pair from count, but it may still exist in the index"""
        del self._count[key]
        self.cache_clear()

    def delete_spans(self, spans: Iterable[str]) -> None:
        """Delete spans from the count table and the index."""
        # Delete from count and index
        for span in spans:
            try:
                index = self._span_index[span]
                del self._count[index]
                del self._span_index[span]
            except KeyError:
                logger.warning(f"Span {span} not in the table during deletion. Ignoring.")
        self.cache_clear()

    def replace(self, other: Self) -> None:
        logger.debug("Replace table")
        if (
            other._span_index != self._span_index
            or other._token_index != self._token_index
        ):
            logger.warning(
                "Replacing the count from a table with different indexes is not safe."
            )
        self.cache_clear()
        self._count = other._count

    @classmethod
    def aggregate(cls, tables: Iterable[Self]) -> Self:
        tmp: Self = tables[0]
        for table in tables[1:]:
            tmp.add(table)
        return tmp

    def replace_aggregate(self, others: Iterable[Self]) -> None:
        tmp = self.aggregate(others)
        self.replace(tmp)

    def __len__(self) -> int:
        return len(self._count)

    @property
    def size(self) -> int:
        """Items in the count table."""
        return len(self)

    @property
    def n_spans(self) -> int:
        """Number of spans in the index."""
        return len(self._span_index)

    @property
    def n_tokens(self) -> int:
        """Number of tokens in the index."""
        return len(self._token_index)

    @classmethod
    def from_corpus(
        cls,
        src_texts: Iterable[list[str]],
        trg_texts: Iterable[str],
        maxlen: int = INF,
        pretokenize: bool = True,
        threshold: int = 0,
        span_list: list[str] = [],
        *,
        quiet: bool = True,
    ) -> Self:
        """
        Initialize a table from a parallel corpus.

        Note:
            If span_list is given, threshold is not used.

        Args:
            src_texts: source tokenized sentences.
            trg_texts: target sentences (not tokenized).
            maxlen: maximum span length.
            pretokenize: split on white spaces.
            threshold: minimum span frequency (if <0: not used).
            span_list: list of spans to keep (if empty: not used).
            quiet: disable progress bar.
        Return:
            Initialized table.
        """
        logger.debug("Initialize from corpus.")
        table = cls()
        # Make filter
        span_filter, n = _build_span_filter(
            trg_texts, span_list, threshold, maxlen, pretokenize, quiet=quiet
        )
        if n == 0:
            try:
                n = len(src_texts)
            except Exception:
                n = 0
        # Populate table and span index
        for src, trg in tqdm(
            zip(src_texts, trg_texts), desc="Table", total=n, disable=quiet
        ):
            spans = iter_spans(trg, maxlen=int(maxlen), pretokenize=pretokenize)            
            spans = filter(span_filter, spans)
            spans = map(table._span_index.get_new, spans)
            tokens = [table._token_index[i] for i in set(src)]
            for span in spans:
                table._count.increase_all(span, tokens)
        return table

    def generate_inv_indexes(self) -> None:
        """
        Generate the dictionaries that maps table indexes to spans and tokens.
        """
        self._inv_token_index = {v: k for k, v in self._token_index.items()}
        self._inv_span_index = {v: k for k, v in dict(iter(self._span_index)).items()}

    @property
    def has_inverted_index(self) -> bool:
        return self._inv_span_index is not None and self._inv_token_index is not None

    def optimize(self) -> None:
        """Convert to contiguous indexes"""
        # Remap old span ind -> new span ind
        new_ind = self._span_index.optimize()
        self._count.reindex(new_ind)
        self.cache_clear()

    def save(self, file: IO[bytes]) -> None:
        """
        This method supports `io.BytesIO` and writable binary files.
        """
        logger.debug("Save")
        with tarfile.open(fileobj=file, mode="w:gz") as tar:
            items: dict[str, SpanIndex|TokenIndex|CountTable] = {
                "_span_index": self._span_index,
                "_token_index": self._token_index,
                "_count": self._count,
            }
            for name, obj in items.items():
                with io.TextIOWrapper(io.BytesIO(), errors="backslashreplace") as buff:
                    obj.serialize(buff)
                    info = tarfile.TarInfo(name)
                    info.size = buff.tell()
                    buff.seek(0)      
                    tar.addfile(info, buff.buffer)

    @classmethod
    def load(cls, file: IO[bytes]) -> Self:
        """
        This method supports `io.BytesIO` and readable binary files.
        """
        logger.debug("Load")
        table = cls()
        with tarfile.open(fileobj=file, mode="r:gz") as tar:
            with io.TextIOWrapper(tar.extractfile("_span_index"), errors="backslashreplace") as buff:
                table._span_index = SpanIndex.deserialize(buff)
            with io.TextIOWrapper(tar.extractfile("_token_index"), errors="backslashreplace") as buff:
                table._token_index = TokenIndex.deserialize(buff)
            with io.TextIOWrapper(tar.extractfile("_count"), errors="backslashreplace") as buff:
                table._count = CountTable.deserialize(buff)
        table.generate_inv_indexes()
        table.lock_alphabet = True        
        return table

    @classmethod
    def load_vocab(cls, file: IO[bytes]) -> dict[str, int]:
        """Load only the vocabulary"""
        with tarfile.open(fileobj=file, mode="r:gz") as tar:
            with io.TextIOWrapper(tar.extractfile("_span_index"), errors="backslashreplace") as buff:
                _span_index = SpanIndex.deserialize(buff)
        return dict(_span_index)

    @functools.lru_cache(MAX_CACHE)
    def sum(self, axis: int | None = None) -> int | dict[int, int]:
        """
        Sum along the given axes.

        Args:
            axis: axes to sum. 0: marginalize the spans, 1: marginalize the tokens.

        Returns:
            The sum as a scalar or as dicitonary {id: count}.

        Notes: cached
        """
        return self._count.sum(axis)
    
    @functools.lru_cache(MAX_CACHE)
    def max(self, axis: int | None = None) -> int | dict[int, int]:
        """
        Max along the given axes.

        Args:
            axis: axes to max. 0: marginalize the spans, 1: marginalize the tokens.

        Returns:
            The max as a scalar or as dicitonary {id: count}.

        Notes: cached
        """
        return self._count.max(axis)
    
    @functools.lru_cache(MAX_CACHE)
    def min(self, axis: int | None = None) -> int | dict[int, int]:
        """
        Min along the given axes.

        Args:
            axis: axes to min. 0: marginalize the spans, 1: marginalize the tokens.

        Returns:
            The min as a scalar or as dicitonary {id: count}.

        Notes: cached
        """
        return self._count.min(axis)
    
    def clear(self) -> None:
        """
        Clear the count table

        Note: this does not reset the index.
        """
        logger.debug("Clear")
        self.cache_clear()
        self._count.clear()

    def cache_clear(self) -> None:
        """
        Clear cached methods.

        Note: we don't clear the alphabet, because we want to keep characters
        that might never appear in loger spans. Recomputing the alphabet would remove those characters.
        """
        # logger.debug("Cache clear")
        self.get.cache_clear()
        self.sum.cache_clear()
        self.__getitem__.cache_clear()        
        self.get_by_id.cache_clear()
        self.get_dict_by_id.cache_clear()
        self.get_token_count.cache_clear()
        self.sum.cache_clear()
        self.max.cache_clear()
        self.min.cache_clear()

    def update_prob_table(self):
        logger.debug("Update counts")
        self._count.update_count()

    @property
    def norm(self) -> int:
        return self._count.norm

    @property
    def lognorm(self) -> float:
        return self._count.lognorm

    @property
    def count_tok(self) -> dict[int, int]:
        return self._count.count_tok

    @property
    def count_span(self) -> dict[int, int]:
        return self._count.count_span

    def increase_all(self, span: str, tokens: list[str], value: float = 1) -> None:
        """
        Increase all the counts with the span and the tokens.
        """
        self.cache_clear()
        if not self.lock_alphabet or set.issuperset(self.alphabet, span):
            s = self._span_index[span]
            t = [self._token_index[i] for i in tokens]
            self._count.increase_all(s, t, value)

    def increase_all_from_id(
        self, span: int, tokens: TokenMask, value: float = 1
    ) -> None:
        self.cache_clear()
        self._count.increase_all(span, tokens, value)


def _build_span_filter(
    trg_texts: Iterable[str],
    span_list: list[str],
    threshold: int,
    maxlen: int = INF,
    pretokenize: bool = True,
    *,
    quiet: bool = True,
) -> tuple[Callable[[str], bool], int]:
    """
    Helper function to select the spans to add to the initial table.

    If span_list is given, threshold is not used.

    Args:
        trg_texts: target sentences.
        span_list: list of spans to keep.
        threshold: minimum span frequency.
        maxlen: maximum span length.
        pretokenize: split on white spaces.
        quiet: disable progress bar.

    Returns:
        Filter that returns true if the input span is to keep.
        Number of sentences in the dataloader.
    """
    logger.debug("Building span filter")
    span_filter = lambda _: True
    n = 0
    if len(span_list) != 0:
        logger.debug("Filter from span list")
        span_list = set(span_list)
        span_filter = lambda x: x in span_list
    elif threshold > 0:
        logger.debug("Filter from threshold")
        # Compute span frequency and alphabet
        span_count = Counter()
        alphabet = set()
        for trg in tqdm(
            trg_texts, desc="Pre-prune count", total=len(trg_texts), disable=quiet
        ):
            n += 1
            alphabet = alphabet.union(trg)
            trg = iter_spans(trg, maxlen=maxlen, minlen=2, pretokenize=pretokenize)
            span_count.update(trg)
        # Create filter
        logger.debug("Creating filter")
        span_set = {s for s, c in span_count.items() if c < threshold}  # REMOVE
        span_filter = lambda x: x not in span_set
        if len(span_count) - len(span_set) < len(span_set):  # Use the shorted list
            logger.debug("Invert filter (keep instead of remove)")
            span_set = {s for s, c in span_count.items() if c > threshold}  # KEEP
            span_set = span_set.union(alphabet)
            span_filter = lambda x: x in span_set
    return span_filter, n
