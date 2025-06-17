import logging
from .. import EMTrainer, ParallelMCTrainer, ForwardBackwardTrainer, MCTrainer, Dataloader, prepare_src, prepare_trg, INF
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

trainer_dict: dict[str, type[EMTrainer]] = {
    "mc": MCTrainer,
    "em": EMTrainer,
}

def train(args):

    logger.debug("Train")

    trg_file = args.trg
    src_file = args.src if args.src is not None else trg_file
    maxlen = args.maxlen if args.maxlen != -1 else INF

    psp = model_dict[args.model](
        vocab_size=args.vocab_size,
        maxlen=maxlen,
        pretokenize=args.pretokenize,
        seed=args.seed,
    )
    trainer = trainer_dict[args.trainer]

    src_dl = Dataloader(file=src_file, fn=prepare_src, preload=args.no_preload)
    trg_dl = Dataloader(file=trg_file, fn=prepare_trg, preload=args.no_preload)

    spt = trainer(
        model=psp,
        src_dl=src_dl,
        trg_dl=trg_dl,
        reduce=args.reduce,
        n_iter=args.iter,
        pretokenize=args.pretokenize,
        char_coverage=args.char_coverage,
        temp_file=args.temp,
        threshold=args.threshold,
        spans=args.spans,
    )

    spt.fit(quiet=args.tqdm)
    with open(args.output, "wb") as f:
        psp.save(f)


def set_parser(train_args):
    train_args.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(model_dict.keys()),
        default="cond-exact",
        help="Type of model.",
    )
    train_args.add_argument(
        "--src",
        type=str,
        help="TOKENIZED sentences in the source language. For Unpaired models: use the same file for --trg.",
    )
    train_args.add_argument(
        "--trg",
        type=str,
        required=True,
        help="PLAIN TEXT sentences in the target language.",
    )
    train_args.add_argument(  # TODO maxlen 10 -> actual length in the loop 9
        "--maxlen",
        "-l",
        type=int,
        default=-1,
        help="Maximum length of the spans in the target language. -1 for unlimited length.",
    )
    train_args.add_argument(
        "--pretokenize",
        "-p",
        action="store_true",
        help="Split the target sentences on white spaces.",
    )
    train_args.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=8000,
        help="Vocabulary size, default 8000.",
    )
    train_args.add_argument(
        "--output", "-o", required=True, type=str, help="Where to save the model"
    )
    train_args.add_argument(
        "--iter",
        "-i",
        default=1,
        type=int,
        help="Maximum number of EM iterations. Stop early if the desired vocabulary size is reached.",
    )
    train_args.add_argument(
        "--sub-iter",
        default=2,
        type=int,
        help="Reduce the vocabulary after this many iterations of the EM algorithm.",
    )
    train_args.add_argument(
        "--reduce",
        "-r",
        default=0.25,
        type=float,
        help="Reduce this portion of vocabulary at each iteration.",
    )
    train_args.add_argument("--seed", "-s", default=0, type=int, help="RNG seed.")
    train_args.add_argument(
        "--char-coverage",
        default=1.0,
        type=float,
        help="Keep the fraction of most frequent character",
    )

    train_args.add_argument(
        "--temp",
        type=str,
        required=False,
        help="Temp. file for intermediate tokenization.",
    )

    train_args.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Remove spans that occour less than the threshold.",
    )

    train_args.add_argument(
        "--spans",
        type=int,
        default=None,
        help="Use a fixed number of spans.",
    )

    train_args.add_argument(
        "--trainer",
        type=str,
        choices=list(trainer_dict.keys()),
        default="mc",
        help="Trainer to use.",
    )
