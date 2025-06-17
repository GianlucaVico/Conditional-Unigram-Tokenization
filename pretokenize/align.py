from collections.abc import Iterator
import tempfile
import os
import sys
import subprocess
import eflomal
from . import pretokenize
import logging

logger = logging.getLogger(__name__)


def compute_priors(src: str, trg: str, prefix: str) -> None:
    """
    Compute eflomal priors and align the input files.

    Args:
        src: source sentences (no need to pretokenize).
        trg: target sentences (no need to pretokenize).
        prefix: prefix for eflomal files.

    Note:
        This function generates {prefix}.[fwd|rev|sym|priors]. It fails if the already exists.
    """
    fwd = f"{prefix}.fwd"
    rev = f"{prefix}.rev"
    sym = f"{prefix}.sym"
    priors = f"{prefix}.priors"

    with tempfile.TemporaryDirectory(dir=".") as tmpdir:
        src_path = os.path.join(tmpdir, src)
        trg_path = os.path.join(tmpdir, trg)
        os.makedirs(os.path.dirname(src_path), exist_ok=True)
        os.makedirs(os.path.dirname(trg_path), exist_ok=True)
        os.makedirs(os.path.dirname(priors), exist_ok=True)
        
        logger.debug("Pretokenize")
        with open(src) as src_i, open(src_path, "w") as src_o:
            for i in pretokenize(src_i):
                src_o.write(i + "\n")
        with open(trg) as trg_i, open(trg_path, "w") as trg_o:
            for i in pretokenize(trg_i):
                trg_o.write(i + "\n")

        logger.debug("Align")
        aligner = eflomal.Aligner()
        with open(src_path) as src_f, open(trg_path) as trg_f:
            aligner.align(src_f, trg_f, links_filename_fwd=fwd, links_filename_rev=rev)

        logger.debug("Symmetrize")
        with open(sym, "w") as sym_file:
            with subprocess.Popen(
                [
                    "atools",
                    "-c",
                    "grow-diag-final-and",
                    "-i",
                    fwd,
                    "-j",
                    rev,
                ],
                text=True,
                stdout=sym_file,
                stderr=sys.stderr,
            ) as p:
                pass

        with open(src_path) as src_f, open(trg_path) as trg_f, open(sym) as fwd_f, open(
            sym
        ) as rev_f, open(priors, "w") as priors_f:
            logger.debug("Compute priors")
            priors_tuple = eflomal.calculate_priors(src_f, trg_f, fwd_f, rev_f)
            logger.debug("Write priors")
            eflomal.write_priors(priors_f, *priors_tuple)


def extract_alignment(
    src: list[str], trg: list[str], alignment: list[str]
) -> Iterator[tuple[str, str]]:
    """
    Convert the eflomal alignment to a list of parallel words.

    Args:
        src: list of source sentences.
        trg: list of target sentences.
        alignment: list of eflomal alignments. (e.g., "0-0 0-1 2-2 3-4 ...", zero indexed)

    Returns:
        Iterator with pairs of words.
    """
    for s, t, a in zip(src, trg, alignment):
        pairs = a.split()
        s_tok = s.split()
        t_tok = t.split()
        for p in pairs:
            si, ti = map(int, p.split("-"))
            yield s_tok[si], t_tok[ti]
