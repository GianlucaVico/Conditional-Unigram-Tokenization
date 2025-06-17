import logging
import fileinput
from .. import utils

logger = logging.getLogger(__name__)


def prepare_dataset(args):
    logger.debug("Prepare")
    if args.type == "src" or args.white:
        f = utils.prepare
    elif args.type == "trg":
        f = utils.prepare_trg

    def gen():
        with fileinput.input(args.input) as fin:
            for i in fin:
                yield f(i)

    utils.to_file(args.output, gen(), lambda x: x)


def set_parser(prepare_args):
    prepare_args.add_argument(
        "type",
        default="trg",
        choices=["src", "trg"],
        help="Process the text as source language or target language (see `utils.py` documentation)",
    )
    prepare_args.add_argument(
        "--input", "-i", default=None, help="Source data, if None use STDIN"
    )
    prepare_args.add_argument(
        "--output", "-o", default=None, help="Output data, if None use STDOUT"
    )
    prepare_args.add_argument(
        "--white", "-w", action="store_true", help="Keep withe spaces"
    )
