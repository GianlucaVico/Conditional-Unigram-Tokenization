"""
Dataloader class for preparing the sentences.

Classes:
    * Dataloader

Methods:
    * identity
"""

from collections.abc import Callable, Iterable, Iterator, Generator
import math
from typing import IO, Any, Self
import logging
import random

logger = logging.getLogger(__name__)


def identity(x: str) -> str:
    """Identity function."""
    return x


class Dataloader[T](Iterable[T]):
    __slots__ = ("_file", "_iter", "_fn", "_preload", "_cache", "_len")

    def __init__(
        self,
        *,
        file: str = None,
        items: Iterable | IO[str] = None,
        fn: Callable[[Any], T] = identity,
        preload: bool = False,
    ) -> None:
        """
        Iterable dataloader that read a file one line at the time.

        Args:
            file: text file, one sentence each line.
            items: create the dataloader from a list of items (file has the priority).
            fn: run this function on each line of the file.
            preload: read and prepare the file immediately and keep it in memory.

        Note:
            `Dataloader.__len__` can be used only if `preload` is True.
        """
        self._file = file
        self._iter = items
        self._fn = fn
        self._preload = preload

        self._cache = None
        self._len = None
        if self._preload:
            self._cache = list(self.generator())

    def __iter__(self) -> Iterator[T]:
        if self._preload:
            return iter(self._cache)
        else:
            return self.generator()

    def generator(self) -> Generator[T]:
        """
        Returns:
            Generator with processed lines.
        """
        if self._file is not None:
            self._iter = open(self._file)
        return map(self._fn, self._iter)

    def generator_dropout(self, dropout_prob: float = 0.5, rng: random.Random = None):
        if rng is None:
            rng = random.Random()

        def drop_gen():
            for i in iter(self):
                if rng.random() > dropout_prob:
                    yield i

        return drop_gen()

    @property
    def preload(self) -> bool:
        """
        Returns:
            If the file is loaded in cache.
        """
        return self._preload

    @property
    def file(self) -> str:
        """
        Returns:
            File path.
        """
        return self._file

    @property
    def fn(self) -> Callable[[Any], T]:
        """
        Returns:
            Function for processing each sentence.
        """
        return self._fn

    def __len__(self) -> int:
        if self._cache is not None:
            return len(self._cache)
        if self._len is not None:
            return self._len
        return 0

    def close(self) -> None:
        if self._file is not None and not self._iter.closed:
            self._iter.close()

    def set_length(self, length: int) -> None:
        if length < 0:
            raise ValueError("Negative length")
        self._len = length

    def split(self, n: int = 1) -> list[Self]:
        """
        Split a dataloader in n dataloaders of roughly equal size.

        Args:
            n: splits
        Returns:
            List of dataloaders.

        Notes:
            The dataloder must be preloaded. If it is not, it will be preloaded.
        """
        if n == 1:
            return [self]
        if not self._preload:
            logger.warning(
                "Splitting requires that the dataloder is preloaded. Preloading."
            )
            self._cache = list(self.generator())
        bucket_size = math.ceil(len(self) / n)
        dls = []
        for i in range(n):
            tmp = self._cache[i * bucket_size : (i + 1) * bucket_size]
            dls.append(self.__class__(items=tmp, preload=True))
        return dls
