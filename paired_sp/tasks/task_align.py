import logging
from tqdm import tqdm
from .. import Dataloader, prepare_src, prepare_trg, to_file
from ..models import *

logger = logging.getLogger(__name__)

align_model_dict: dict[str, type[PairedSPModel]] = {
    "max": PairedSPMaxModel,
    "sum": PairedSPSumModel,
    "max-unpaired": PairedSPMaxModel,
    "sum-unpaired": PairedSPSumModel,
    "cond-exact": PairedSPExactCondModel,
    "pmi": PairedSPPMI,
    "cpmi": PairedSPCPMI,
}


def align(args):
    logger.debug("Align")
    trg_file = args.trg
    src_file = args.src if args.src is not None else trg_file

    with open(args.model_file, "rb") as f:
        psp = align_model_dict[args.model].load(f)

    if args.tokenize:
        src_dl = Dataloader(file=src_file, fn=prepare_src, preload=args.no_preload)
        trg_dl = Dataloader(file=trg_file, fn=prepare_trg, preload=args.no_preload)

        def tokenize():
            for s, t in tqdm(zip(src_dl, trg_dl), desc="Tokenize", disable=args.tqdm):
                yield psp.pair_tokenize(s, t)

        src_dl = Dataloader(file=src_file, fn=prepare_src, preload=args.no_preload)
        trg_dl = Dataloader(items=tokenize())
    else:
        src_dl = Dataloader(file=src_file, fn=prepare_src, preload=args.no_preload)
        trg_dl = Dataloader(file=trg_file, fn=prepare_src, preload=args.no_preload)

    def gen():
        for s, t in tqdm(zip(src_dl, trg_dl), desc="Align", disable=args.tqdm):
            a = psp.align(s, t)
            a = [f"{i}-{j}-{k}" for i, j, k in a]
            yield a

    to_file(args.output, gen(), lambda x: " ".join(x))


def set_parser(align_args):
    align_args.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(align_model_dict.keys()),
        help="Type of model.",
    )
    align_args.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="tar.gz file that contains the model.",
    )
    align_args.add_argument(
        "--src",
        type=str,
        required=True,
        help="TOKENIZED sentences in the source language. For Unpaired models: use the same file for --trg.",
    )
    align_args.add_argument(
        "--trg",
        type=str,
        required=True,
        help="PLAIN TEXT sentences in the target language.",  # TODO update description
    )
    align_args.add_argument(
        "--output",
        "-o",
        type=str,
        help="Where to save the tokenized data. Default STDOUT.",
    )

    align_args.add_argument(
        "--tokenize",
        action="store_true",
        help="Tokenize the input target file.",
    )
