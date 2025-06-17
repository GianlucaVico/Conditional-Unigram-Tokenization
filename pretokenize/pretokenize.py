from collections.abc import Iterator
from paired_sp import prepare


def pretokenize(lines: list[str]) -> Iterator[str]:
    "Iterator that returns clean lines."
    for line in lines:
        yield prepare(line)
