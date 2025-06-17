"""
Train a SentencePiece tokenizer and use its vocabulary as an initial list of spans
"""

# TODO fix logs/SP version
from collections.abc import Iterator
import io
import random
from typing import Literal, Self
import sentencepiece as spm
import yaml
import re

from .dataloader import Dataloader
import logging

logger = logging.getLogger(__name__)


class SentencePieceSpans:
    def __init__(
        self,
        vocab_size: int = 1_000_000,
        character_coverage: float = 1.0,
        model_type: Literal["bpe", "unigram"] = "bpe",
    ) -> None:
        """
        List of spans from a SentencePiece model.

        Args:
            vocab_size: number of spans, used when training a new model.
            character_coverage: SentencePiece character coverage.
            model_type: bpe or unigram model.
            use_iterator: read the sentences from an iterator instead of files (this also forces keep_model to false)
        """
        self._vocab_size = vocab_size
        self._coverage = character_coverage
        self._model_type = model_type

        self.spans: list[str] = []

    def train(self, inputs: Dataloader[str], runs: int = 1) -> None:
        """
        Train a new SentencePiece model and load its vocabulary.

        Args:
            inputs: iterator with training examples.
            runs: take the union of multiple runs
        """
        sp = []
        try:
            spm.SetMinLogLevel(1000)
        except AttributeError:
            logger.warning("Old SentencePiece version, verbose output.")
        for _ in range(runs):
            try:
                rng = random.Random()  # Use this to clone the generator
                rng_state = rng.getstate()
                inp = inputs.generator_dropout(rng=rng)
                buff = io.BytesIO()
                spm.SentencePieceTrainer.train(
                    sentence_iterator=inp,
                    vocab_size=self._vocab_size,
                    character_coverage=self._coverage,
                    model_type=self._model_type,
                    model_writer=buff,
                    minloglevel=logger.root.level,
                )
            except RuntimeError as err:
                # Vocabulary is too large
                err_msg = str(err)
                regex_match = re.search(r"(?<=<=) \d*", err_msg)
                if regex_match is not None:
                    # Fix
                    old_size = self._vocab_size
                    new_size = int(regex_match.group())
                    logger.warning(
                        f"Vocabulary size for SP spans is too high ({old_size}). Set it to {new_size}."
                    )
                    # Retry
                    rng = random.Random()
                    rng.setstate(rng_state)
                    inp = inputs.generator_dropout(rng=rng)
                    buff = io.BytesIO()
                    spm.SentencePieceTrainer.train(
                        sentence_iterator=inp,
                        vocab_size=new_size,
                        character_coverage=self._coverage,
                        model_type=self._model_type,
                        model_writer=buff,
                        minloglevel=logger.root.level,
                    )
                else:
                    raise err

            tmp = spm.SentencePieceProcessor(model_proto=buff.getvalue())
            sp.append(set([tmp.id_to_piece(id) for id in range(tmp.get_piece_size())]))
            buff.close()
        sp = set.union(*sp)
        self.spans = list(sp)

    def __iter__(self) -> Iterator[str]:
        return iter(self.spans)

    def __len__(self) -> int:
        return len(self.spans)

    def __contains__(self, item: str) -> bool:
        return item in self.spans

    @classmethod
    def fram_yaml(self, vocab: str) -> Self:
        """
        Load the vocabulary from a yaml file.
        The file contains a dictionary with the spans and their ids (the ids are not used).

        Args:
            vocab: path to the yaml file.
        """
        # token: id
        d = {}
        with open(vocab) as v:
            d: dict[str, int] = yaml.load(v, yaml.Loader)

        sps = SentencePieceSpans(len(d))
        sps.spans = list(d.keys())
        return sps
