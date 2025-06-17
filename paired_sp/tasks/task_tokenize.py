import logging

from tqdm import tqdm
from .. import Dataloader, prepare_src, prepare_trg, prepare_align, to_file
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


def tokenize(args):
    logger.debug("Tokenize")
    trg_file = args.trg

    if args.src is None:
        src_file = trg_file
        src_dl = Dataloader(file=src_file, fn=lambda x: "", preload=args.no_preload)
    else:
        src_file = args.src
        src_dl = Dataloader(file=src_file, fn=prepare_src, preload=args.no_preload)
    if args.alignment is None:
        def align_gen():
            while True:
                yield None
        align_dl = align_gen()
    else:
        align_dl = Dataloader(file=args.alignment, fn=prepare_align, preload=args.no_preload)

    trg_dl = Dataloader(file=trg_file, fn=prepare_trg, preload=args.no_preload)

    with open(args.model_file, "rb") as f:
        psp = model_dict[args.model].load(f)
    
    def gen():
        for s, t, a in tqdm(zip(src_dl, trg_dl, align_dl), desc="Tokenize", disable=args.tqdm):
            yield psp.pair_tokenize(s, t, a)

    to_file(args.output, gen(), lambda x: " ".join(x))


def set_parser(tokenize_args):
    tokenize_args.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(model_dict.keys()),
        default="cond-exact",
        help="Type of model.",
    )
    tokenize_args.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="tar.gz file that contains the model.",
    )
    tokenize_args.add_argument(
        "--src",
        type=str,
        help="TOKENIZED sentences in the source language. For Unpaired models: use the same file for --trg.",
    )
    tokenize_args.add_argument(
        "--trg",
        type=str,
        required=True,
        help="PLAIN TEXT sentences in the target language.",
    )
    tokenize_args.add_argument(
        "--alignment",
        type=str,
        help="Eflomal alignment."
    )
    tokenize_args.add_argument(
        "--output",
        "-o",
        type=str,
        help="Where to save the tokenized data. Default STDOUT.",
    )
