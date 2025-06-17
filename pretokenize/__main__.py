import argparse
from collections.abc import Iterator
import os
from typing import Literal
from . import compute_priors, extract_alignment, pretokenize, extract_phrases

import logging

try:
    from rich.logging import RichHandler

    logging.basicConfig(
        level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
except ImportError:
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[logging.StreamHandler()],
    )

logger = logging.getLogger(__name__)

def extract_to_file(it: Iterator[tuple[str, str]], src_out: str, trg_out: str) -> None:
    """
    Write an iterator of parallel sentences to file.

    Args:
        it: iterator with pairs of strings
        src_out: path for the first string in the pair.
        trg_out: path for the second string in the pair.
    """
    with open(src_out, "w") as src_f, open(trg_out, "w") as trg_f:
        for src_t, trg_t in it:
            src_f.write(src_t + "\n")
            trg_f.write(trg_t + "\n")


def make_files(
    src_input: str,
    trg_input: str,
    src_output: str,
    trg_output: str,
    type: Literal["align", "phrases"] = "align",
    prefix: str | None = None,
) -> None:
    """
    Extract the phrases or the alignements and save them.

    Args:
        src_input: path to source sentences.
        trg_input: path to target sentences.
        src_output: path to source phrases.
        trg_output: path to target phrases.
        type: "align" for word alignment, "phrases" for phraes tables
        prefix: prefix for eflomal intermediate files. If None, it uses src_input.
    """
    if prefix is None:
        prefix = src_input
    alignment_file = prefix + ".sym"
    if not os.path.exists(alignment_file):
        compute_priors(src_input, trg_input, prefix)

    with open(src_input) as src, open(trg_input) as trg, open(
        alignment_file
    ) as alignment:
        if type == "align":
            it = extract_alignment(pretokenize(src), pretokenize(trg), alignment)
        elif type == "phrases":
            it = extract_phrases(pretokenize(src), pretokenize(trg), alignment)
        extract_to_file(it, src_output, trg_output)


if __name__ == "__main__":
    # TODO pretokenize only once
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-input", "-si", help="Source text.")
    parser.add_argument("--trg-input", "-ti", help="Target text.")
    parser.add_argument("--prefix", "-p", help="Prefix for eflomal files.")
    parser.add_argument("--src-output", "-so", help="Source text.")
    parser.add_argument("--trg-output", "-to", help="Target text.")
    parser.add_argument(
        "--type",
        "-t",
        choices=["align", "phrases"],
        default="align",
        help="'phrases to extract parallel phrases; 'align' to extract aligned words.",
    )
    logger.root.setLevel(logging.WARNING)
    args = parser.parse_args()
    make_files(
        args.src_input,
        args.trg_input,
        args.src_output,
        args.trg_output,
        args.type,
        args.prefix,
    )
