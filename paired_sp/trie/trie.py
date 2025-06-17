"""
Data structures for storing and managing the count table.
"""

import functools
from typing import IO, Self
from collections.abc import Callable, Iterable
import ipdb
import numpy as np
import logging
import json
from ..utils import TokenMask, empty_max, empty_min

logger = logging.getLogger(__name__)

MAX_CACHE = 128


class SpanIndex:
    def __init__(self) -> None:
        """
        Trie that store the indexes for the character spans.

        Implementation notes:
            Strings are stored in a trie that is made of nested dictionaries.
            The entry None stores the index of the string build from the current path.
        """
        logger.debug("Create SpanIndex")
        self._root = {}
        self._size = 0

    @functools.lru_cache(MAX_CACHE)
    def __getitem__(self, key: str) -> int:
        """
        Args:
            key: span to retrieve

        Returns:
            Index

        Notes: cached
        """
        node = self._root
        for s in key:
            node = node[s]
        return node[None]

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __setitem__(self, key: str, value: int) -> None:
        """
        Insert a string and assign an index.

        Args:
            key: string
            value: index

        Notes:
            This method is intend for internal use since this method does not
            check for duplicated indexes.
        """
        self.cache_clear()
        node = self._root
        for s in key:
            child = node.get(s)
            # Add node
            if child is None:
                child = {}
                node[s] = child
            node = child
        if None not in node:
            self._size += 1
        node[None] = value

    def __len__(self) -> int:
        return self._size

    def __delitem__(self, key: str) -> None:
        node = self._root
        for s in key:
            child = node.get(s)
            # Add node
            if child is None:
                return
            node = child
        if None in node:
            del node[None]
            self.cache_clear()
            self._size -= 1

    def __iter__(self) -> Iterable[tuple[str, int]]:
        logger.debug("Iterate span index")

        def visit_note(n: dict, path: list[str]):
            for k in n:
                if k is None:
                    yield ("".join(path), n[k])
                else:
                    path.append(k)
                    yield from visit_note(n[k], path)
                    path.pop()

        return visit_note(self._root, [])

    def optimize(self) -> dict[int, int]:
        """
        Reindex the spans to have continuous indexes.

        Returns:
            Dictionary that maps the old indexes to the new indexes.
        """
        logger.debug("Optimize span index")
        self.prune()
        new_index = {}

        def visit_node(n: dict):
            for k, child in n.items():
                if k is None:
                    n[k] = len(new_index)
                    new_index[child] = len(new_index)
                else:
                    visit_node(n[k])

        visit_node(self._root)
        return new_index

    def prune(self) -> None:
        """Remove all empty nodes"""
        logger.debug("Prune span index")
        self.cache_clear()

        def check_node(n: dict):
            if n is None or isinstance(n, int):
                return
            ks = list(n.keys())
            for k in ks:
                if n[k] is not None and not isinstance(n[k], int):
                    check_node(n[k])
                    if len(n[k]) == 0:
                        del n[k]

        check_node(self._root)

    @functools.lru_cache(MAX_CACHE)
    def get(self, key: str, default: int | None = None) -> int | None:
        """
        Get an index or the default value.

        Args:
            key: character span
            default: value if the span is not in the trie
        Return:
            Index
        """
        node = self._root
        for s in key:
            node = node.get(s)
            if node is None:
                return default
        ret = node.get(None)
        if ret is None:
            return default
        return ret

    @functools.lru_cache(1)
    def todict(self) -> dict[str, int]:
        """Flatten the trie"""
        return dict(iter(self))

    def update(self, span: str) -> bool:
        """
        Add the span if it is not in the index

        Args:
            span: span to add
        Returns:
            True if the spans is added, False otherwise
        """
        node = self._root
        for s in span:
            child = node.get(s)
            # Add node
            if child is None:
                child = {}
                node[s] = child
            node = child
        if None not in node:
            node[None] = self._size
            self._size += 1
            self.cache_clear()
            return True
        return False

    def get_new(self, span: str) -> int:
        """Retrieve the index if it exists, otherwise add it"""
        self.update(span)  # Add it if new
        return self[span]

    def extend(self, spans: Iterable[str]) -> list[bool]:
        """Update but for a list of strings"""
        return [self.update(i) for i in spans]

    def __repr__(self) -> str:
        return repr(self._root)

    def __str__(self) -> str:
        return str(self._root)

    @property
    def alphabet(self) -> set[str]:
        chars = set()

        def visit(n: dict, chars: set):
            chars.update(n.keys())
            for v in n.values():
                if isinstance(v, dict):
                    visit(v, chars)

        visit(self._root, chars)
        chars.discard(None)
        return chars

    def cache_clear(self) -> None:
        """Clear the cached values."""
        self.todict.cache_clear()
        self.__getitem__.cache_clear()
        self.get.cache_clear()

    def __hash__(self) -> int:
        return hash(self._root.values())

    def __eq__(self, value: Self) -> bool:
        return self.todict() == value.todict()

    def serialize(self, file: IO[str] | None) -> str | None:
        """
        Serialize the trie to a json file.

        Args:
            file: file to write the trie

        Returns:
            JSON string if file is None, None otherwise.
        """
        data = self.todict()
        if file is None:
            return json.dumps(data, indent=2)
        json.dump(data, file, indent=2)
    
    @classmethod
    def deserialize(cls, file: IO[str]) -> Self:
        """
        Deserialize a trie from a json file.

        Args:
            file: file to read the trie

        Returns:
            SpanIndex
        """
        data: dict[str, int] = json.load(file)
        trie = cls()
        for k, v in data.items():
            trie[k] = v
        return trie

class TokenIndex(dict):
    """
    Dictionary that maps tokens to their index.
    If a token is not in the dictionary it is added immediatly.
    """

    def __missing__(self, key: str) -> int:
        tmp = len(self)
        self[key] = tmp
        return tmp

    def append(self, key: str) -> None:
        self[key]

    def serialize(self, file: IO[str] | None) -> str | None:
        """
        Args:
            file: file to write the trie

        Returns:
            JSON string if file is None, None otherwise.
        """        
        if file is None:
            return json.dumps(self, indent=2)
        json.dump(self, file, indent=2)
    
    @classmethod
    def deserialize(cls, file: IO[str]) -> Self:
        """
        Args:
            file: file to read the trie

        Returns:
            SpanIndex
        """
        data: dict[str, int] = json.load(file)
        index = cls()
        for k, v in data.items():
            index[k] = v
        return index

class CountTable(dict):
    def __init__(self) -> None:
        """
        Dictionary of dictionary that stores the counts of span indexes and token indexes.

        Note:
            Pickle will try to load this class as a dictionary.
        """
        self._size = 0
        # Pre compute some sums
        self._norm = 0
        self._lognorm = 0
        self._count_t = None
        self._count_s = None

    def __getitem__(self, key: tuple[int, int] | int) -> float | dict[int, float]:
        """
        Args:
            key: pair of span index and token index, or a single span index

        Returns:
            Count of the pair or dictionary with {token index: count}.
        """
        if isinstance(key, int):
            return super().__getitem__(key)
        else:
            return super().__getitem__(key[0])[key[1]]

    def get(self, key: int | tuple[int, int], default: int | None = 0) -> float | dict[int, float]:
        """
        Args:
            key: pair of span index and token index, or a single span index
            default: returned if the span is not in the table.

        Returns:
            Count of the pair or dictionary with {token index: count}.
        """
        if isinstance(key, int):
            return super().get(key, default)
        tmp = self.get(key[0], None)
        if tmp is None:
            return default
        return tmp.get(key[1], default)

    def __setitem__(
        self, key: tuple[int, int] | int, value: float | dict[int, float]
    ) -> None:
        """
        Insert an entry or a "row" in the table.

        Args:
            key: pair of span and token indexes, or a single span index.
            value: count for the pair, or counts for the tokens as a dictionary {token index: count}
        """
        if isinstance(key, int):
            tmp = self.get(key, None)
            if "_size" not in self.__dict__:
                self._size = 0
            if tmp is None:
                tmp = {}
                super().__setitem__(key, tmp)            
            self._size -= len(tmp)
            tmp.update(value)
            self._size += len(tmp)
        else:
            tmp = self.get(key[0], None)
            if tmp is None:
                tmp = {}
                self.__setitem__(key[0], tmp)
            # Update size (query only once)
            self._size -= len(tmp)
            tmp[key[1]] = value
            self._size += len(tmp)

    def __delitem__(self, key: int | tuple[int, int]) -> None:
        """Remove an entry or a "row" from the table."""
        if isinstance(key, int):
            self._size -= len(self[key])
            super().__delitem__(key)
        else:
            tmp = self.get(key[0], None)
            if tmp is not None:
                del tmp[key[1]]
                self._size -= 1

    def __len__(self) -> int:
        return self._size
    
    def _vector_fn(self, fn: Callable[[Iterable[float]], float], axis: int | None = None) -> float | dict[int, float]:
        """
        Apply the function along the given axes.

        Args:
            fn: function to apply. It takes the row values or the list with [partial_output, new_input]
            axis: axes to sum. 0: marginalize the spans, 1: marginalize the tokens.

        Returns:
            Output of the function as float or dictionary {id: value}.
        """
        if axis is None:
            return fn(fn(i.values()) for i in self.values())
        elif axis == 0:
            tmp = {}
            for tok_count in self.values():
                for tok, count in tok_count.items():
                    tmp[tok] = fn([tmp.get(tok, 0), count])
            return tmp
        elif axis == 1:
            return {span: fn(tok_count.values()) for span, tok_count in self.items()}

    def sum(self, axis: int | None = None) -> float | dict[int, float]:
        """Sum along the given axes."""
        return self._vector_fn(sum, axis=axis)
    
    def min(self, axis: int | None = None) -> float | dict[int, float]:
        """Min along the given axes."""        
        return self._vector_fn(empty_min, axis=axis)
    
    def max(self, axis: int | None = None) -> float | dict[int, float]:
        """Max along the given axes."""
        return self._vector_fn(empty_max, axis=axis)

    def add(self, other: dict[int, dict[int, int]] | Self | int) -> None:
        """
        Args:
            other: count table or number to add.
        """
        if isinstance(other, int):
            for _, tok_count in self.items():
                for tok in tok_count:
                    tok_count[tok] += other
        else:
            for span, tok_count in other.items():
                tmp = self.get(span, None)
                if tmp is None:
                    tmp = {}
                    self[span] = tok_count
                    self._size += len(tok_count)
                else:
                    for tok, count in tok_count.items():
                        old = tmp.get(tok)
                        if old is None:
                            self._size += 1
                            tmp[tok] = count
                        else:
                            tmp[tok] = old + count
        self.update_count()

    def __iadd__(self, other: dict[int, dict[int, int]] | Self | int) -> Self:
        self.add(other)
        return self

    def __imul__(self, other: int) -> Self:
        for _, tok_count in self.items():
            for tok in tok_count:
                tok_count[tok] *= other
        self.update_count()
        return self

    def __itruediv__(self, other: int) -> Self:
        return self.__imul__(1 / other)

    def __floordiv__(self, other: int) -> Self:
        for _, tok_count in self.items():
            for tok in tok_count:
                tok_count[tok] //= other
        self.update_count()
        return self

    def clear(self) -> None:
        """Clear the count table"""
        logger.debug("Count table clear.")
        super().clear()
        self._norm = 0
        self._lognorm = -np.inf
        self._count_s = None
        self._count_t = None
        self._size = 0

    def sparsity(self) -> float:
        """
        Compute the sparsity of the count table.
        0: dense, 1: empty

        """
        s = len(self._count_s)
        t = len(self._count_t)
        
        return 1 - self._size / (s * t)

    def reindex(self, map_: dict[int, int]) -> None:
        """
        Change the span index

        Args:
            map_: old to new index
        """
        logger.debug("Count table reindex")
        tmp = {}
        for span, tok_count in self.items():
            new_span = map_[span]
            tmp[new_span] = tok_count
        self.clear()
        for span, tok_count in tmp.items():
            super().__setitem__(span, tok_count)
            self._size += len(tok_count)
        self.update_count()

    def increase_all(self, span: int, tokens: TokenMask, value: float = 1) -> None:
        """
        Increase the count of all the tokens by value.
        """
        count = self.get(span, None)
        if count is None:
            count = {}
            super().__setitem__(span, count)
        self._size -= len(count)
        for i in tokens:
            count[i] = count.get(i, 0) + value
        self._size += len(count)

    def update_count(self) -> None:
        self._norm = self.sum()
        self._lognorm = np.log(self._norm)
        self._count_s = self.sum(1)
        self._count_t = self.sum(0)

    @property
    def norm(self) -> int:
        """Total value of the table"""
        return self._norm

    @property
    def lognorm(self) -> float:
        return self._lognorm

    @property
    def count_tok(self) -> dict[int, int]:
        return self._count_t

    @property
    def count_span(self) -> dict[int, int]:
        return self._count_s

    def serialize(self, file: IO[str] | None) -> str | None:
        """
        Args:
            file: file to write the trie

        Returns:
            JSON string if file is None, None otherwise.
        """
        if file is None:
            return json.dumps(self, indent=2)
        json.dump(self, file, indent=2)
    
    @classmethod
    def deserialize(cls, file: IO[str]) -> Self:
        """
        Deserialize a trie from a json file.

        Args:
            file: file to read the trie

        Returns:
            SpanIndex
        """
        data: dict[str, dict[str, int]] = json.load(file)
        count = cls()
        for span_s, row_s in data.items():
            row = {int(tok): count for tok, count in row_s.items()}
            count[int(span_s)] = row
        count.update_count()
        return count
