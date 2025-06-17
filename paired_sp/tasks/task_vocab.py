import logging
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


def vocab(args):
    logger.debug("Vocab")
    with open(args.model_file, "rb") as f:
        psp_vocab = model_dict[args.model].load_vocab(f)
    
    with open(args.output, "w") as out:
        out.write(PairedSPModel.export_vocab_static(psp_vocab))


def set_parser(vocab_args):
    vocab_args.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(model_dict.keys()),
        help="Type of model.",
    )
    vocab_args.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="tar.gz file that contains the model.",
    )

    vocab_args.add_argument("-o", "--output", type=str, required=True, help="Output file (yaml).")
