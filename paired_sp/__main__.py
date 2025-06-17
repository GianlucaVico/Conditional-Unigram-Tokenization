from . import tasks
from . import *
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="Paired SentencePiece")
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--no-preload", action="store_false", help="Preload in memory all the datasets."
    )
    base_parser.add_argument(
        "--tqdm", action="store_false", help="Show tqdm progress bars."
    )
    base_parser.add_argument("--debug", action="store_true", help="Logger debug")
    base_parser.add_argument("--info", action="store_true", help="Logger info")

    subparsers = parser.add_subparsers(title="Tasks", required=True, dest="task")

    ### TRAIN ###

    train_args = subparsers.add_parser(
        "train", help="Train a new model", parents=[base_parser]
    )
    tasks.task_train.set_parser(train_args)

    ### TOKENIZE ###

    tokenize_args = subparsers.add_parser(
        "tokenize", help="Tokenize paired sentences", parents=[base_parser]
    )
    tasks.task_tokenize.set_parser(tokenize_args)

    ### ALIGN ###

    align_args = subparsers.add_parser(
        "align", help="Aligne paired sentences", parents=[base_parser]
    )
    tasks.task_align.set_parser(align_args)

    ### VOCAB ###

    vocab_args = subparsers.add_parser(
        "vocab", help="Export the vocabulary.", parents=[base_parser]
    )
    tasks.task_vocab.set_parser(vocab_args)

    ### EXTRACT ###

    extract_args = subparsers.add_parser(
        "extract", help="Extract span-token-score triples.", parents=[base_parser]
    )
    tasks.task_extract.set_parser(extract_args)

    ### PREPARE SRC/TRG  ###
    prepare_args = subparsers.add_parser(
        "prepare", help="Prepare the scr or target text", parents=[base_parser]
    )
    tasks.task_prepare.set_parser(prepare_args)

    args = parser.parse_args()
    if args.debug:
        logger.root.setLevel(logging.DEBUG)
    elif args.info:
        logger.root.setLevel(logging.INFO)
    else:
        logger.root.setLevel(logging.WARNING)

    logger.debug(args)
    if args.task == "train":
        tasks.task_train.train(args)
    elif args.task == "tokenize":
        tasks.task_tokenize.tokenize(args)
    elif args.task == "align":
        tasks.task_align.align(args)
    elif args.task == "vocab":
        tasks.task_vocab.vocab(args)
    elif args.task == "extract":
        tasks.task_extract.extract(args)
    elif args.task == "prepare":
        tasks.task_prepare.prepare_dataset(args)
