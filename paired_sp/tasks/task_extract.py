import logging
from itertools import chain
from .. import to_file
from ..models import *

logger = logging.getLogger(__name__)

model_dict: dict[str, type[PairedSPModel]] = {
    "max": PairedSPMaxModel,
    "sum": PairedSPSumModel,
    "max-unpaired": UnpairedSPMaxModel,
    "sum-unpaired": UnpairedSPSumModel,
    "cond-exact": PairedSPExactCondModel,
    "pmi": PairedSPPMI,
    "cpmi": PairedSPCPMI,
}


def extract(args):
    logger.debug("Extract")
    with open(args.model_file, "rb") as f:
        psp = model_dict[args.model].load(f)
        # spm.count_table.densify()

    pairs = psp.extract_pairs(args.k)
    header = [("span", "token", "score")]

    to_file(args.output, chain(header, pairs), lambda x: "\t".join(map(str, x)))


def set_parser(extract_args):
    extract_args.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(model_dict.keys()),
        help="Type of model.",
    )
    extract_args.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="tar.gz file that contains the model.",
    )
    extract_args.add_argument(
        "-k",
        type=int,
        default=1,
        help="Select the best top k alignments. -1 select all the spans with all the tokens.",
    )
    extract_args.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file (tsv file with 'span', 'token', 'score').",
    )
