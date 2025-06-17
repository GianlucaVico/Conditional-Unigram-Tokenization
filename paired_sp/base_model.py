"""
Module for the paired SP trainer and base SP model.

Classes:
    * PairedSPTrainer
    * PairedSPModel

Methods:
    * load
"""

from __future__ import annotations
import io
import json
import time
import pickle
import tarfile
import ipdb
import yaml
from typing import IO, Self
import numpy as np
import numpy.typing as npt
import logging

from .trie import Table
from .dataloader import Dataloader
from .utils import WHITE_CHAR, INF

logger = logging.getLogger(__name__)


class PairedSPModel:
    def __init__(
        self,
        vocab_size: int,
        maxlen: int = INF,
        pretokenize: bool = False,
        seed: int = 0,
    ) -> None:
        """
        Base class for the PairedSP models.

        Args:
            vocab_size: desired vocabulary size.
            maxlen: maximum length of the spans (`numpy.inf` for no maximum length).
            pretokenize: spans can't corss word boundaries.
            seed: seed for the random number generator.
        """
        self.count_table: Table = None
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.pretokenize = pretokenize
        self.seed = 0
        self.rng = np.random.default_rng(seed)

    def score_pair(self, src_ids: npt.ArrayLike, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Implemented by the subclasses.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Score.
        """
        raise NotImplementedError()

    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:
        """
        Align a sentence in the source language and a sentence in the target language.

        Implemented by the subclasses.

        Args:
            src: source tokens.
            trg: target tokens.

        Returns:
            Alignment as list of tuples. The tuple contains the index of the source token,
            the index of the target token and the score of the aligned pair.
        """
        raise NotImplementedError()

    def extract_pairs(self, k: int | None = 1) -> list[tuple[str, str, float]]:
        """
        Find the corresponding spans and tokens and score them.

        Args:
            k: top-k parameter. If None, use the entire table

        Returns:
            List of (span, token, score)

        Note: subclasses should assume that the table uses a sparse matrix and do the extraction row by row.
        """
        raise NotImplementedError()

    def init_table(
        self,
        src_texts: Dataloader[list[str]],
        trg_texts: Dataloader[str],
        pretokenize: bool = True,
        threshold: int = 0,
        span_list: list[str] = [],
        *,
        quiet: bool = True,
    ) -> None:
        """
        Initialize the cooccurence table and some values obtained from it.

        Args:
            src_texts: dataloader with tokenized sentences in the source language.
            trg_texts: dataloader with sentences in the target languages.
            pretokenize: spans do not cross word boundaries.
            threshold: remove spans that occour less than the threshold.
            quiet: disable tqdm.
        """
        # Initialize count table
        self.count_table = Table.from_corpus(
            src_texts,
            trg_texts,
            self.maxlen,
            pretokenize,
            threshold,
            span_list,
            quiet=quiet,
        )

        # Update actual maxlen
        self.maxlen = max([len(i) for i in self.count_table.spans])
        self.update_prob_table()

    def update_prob_table(self) -> None:
        """Update norm and lognorm. Subclasses can update other stats obtained from the table."""
        self.count_table.update_prob_table()

    # @np.errstate(divide="ignore")
    def paired_forward(self, src: list[str], trg: str) -> tuple[float, list[int]]:
        """
        Compute scores and span size for tokenization.

        Args:
            src: tokens in the source language.
            trg: target sentece.

        Returns:
            Loss and list of best span sizes for each position. E.g. if best[7]
            is 3, then the best way to reach position 7 is with a span of size 3.
        """
        count_table = self.count_table
        l = len(trg)

        # Log prob at every position
        best = [-np.inf] * (l + 1)
        best[0] = 0

        # Size of the last span at every position
        sizes = [None] * (l + 1)
        token_mask = count_table.get_token_mask(src)
        # TODO check correctness
        for i in range(1, l + 1):  # Current character
            if trg[i - 1] not in count_table.alphabet:  # UNK character, skip
                sizes[i] = 1
                best[i] = 0
                logger.warning(f"Encountered unknown character '{trg[i-1]}' in {trg}")
                continue
            # for j in range(max(i - self.maxlen, 0), i):  # Start of the span
            for j in range(i - 1, max(i - self.maxlen, -1), -1):  # End of the span
                span = trg[j:i]
                span_id = count_table.get_span_id(span)
                if span_id is not None:
                    score = self.score_pair(token_mask, span_id)
                    if not np.isfinite(score):
                        score = count_table.min()
                    if best[j] + score > best[i]:
                        best[i] = best[j] + score
                        sizes[i] = len(span)
                if trg[j] == WHITE_CHAR and self.pretokenize:
                    break
            if sizes[i] is None:
                sizes[i] = 1
                best[i] = 0
                logger.warning(f"Encountered token with unknow score '{trg[i-1]} in {trg} / {src}'.")
        loss = best[-1]  # Log prob at the end of the sequence
        return loss, sizes

    def paired_backward(self, trg: str, sizes: list[int]) -> list[str]:
        """
        Args:
            trg: target sentence.
            sizes: span size for each position.

        Returns:
            Optimal list of spans.
        """
        i = len(sizes)
        tokens = []
        while i > 1:
            # Start of the span
            next_i = i - sizes[i - 1]
            # Add span
            tokens.append(trg[next_i - 1 : i - 1])
            i = next_i
        return tokens[::-1]

    def pair_tokenize(self, src: list[str], trg: str, alignment: list[tuple[int, int]] | None = None) -> list[str]:
        """
        Tokenize a string in the target language given the corresponding
        tokenized sentence in the source language.

        Args:
            src: tokenized source sentence.
            trg: target sentence.

        Returns:
            List of tokens.
        """
        if alignment is None:
            _, p = self.paired_forward(src, trg)
            tokenization = self.paired_backward(trg, p)
            return tokenization
        else: # TODO THIS NEEDS SUBWORD EFLOMAL ALIGNMENT, NOT WORD ALIGNMENT
            trg_words = trg.split(WHITE_CHAR)
            trg_words = [f"{WHITE_CHAR}{i}" for i in trg_words if i != ""]
            tokenization = []
            for i, trg_word in enumerate(trg_words):
                src_toks = [pair[0] for pair in alignment if pair[1] == i]
                
                if len(src_toks) == 0:
                    src_toks = src
                _, p = self.paired_forward(src_toks, trg_word)
                tokenization.extend(self.paired_backward(trg_word, p))
            return tokenization

    @classmethod
    def from_model(cls, other: PairedSPModel) -> Self:
        """
        Transform a paired sp model into a paired sp model of this type.
        THe cooccurence table is the same.

        Args:
            other: original model.

        Returns:
            Same model with this type.
        """
        new_ = cls(other.vocab_size, other.maxlen, other.pretokenize, 0)

        new_.count_table = other.count_table
        new_.rng = other.rng
        new_.update_prob_table()

        return new_

    def encode(
        self,
        trg: list[str],
        unk_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 3,
        pad_length: int | None = None,
    ) -> list[int]:
        """
        Encode a tokenized sentence.
        This method can add some special tokens and uses bytes for OOV (i.e. the UNK token
        should never appear).

        Args:
            trg: tokenized target sentence.
            unk_id: id for unknown tokens (-1: not set).
            bos_id: id for begin of sequence (-1: not set).
            eos_id: id for end of sequence (-1: not set).
            pad_id: id for padding (-1: not set).
            pad_length: pad the sequence up to this length (None if not used).

        Returns:
            List of ids.
        """
        assert not (
            pad_id < 0 and pad_length is not None
        ), "If pad is used pad id must be >= 0"

        ids = [] if bos_id < 0 else [bos_id]

        keys = self.count_table._span_index.todict()
        reserved = max(-1, unk_id, bos_id, eos_id, pad_id)
        shift = 256 + reserved + 1
        for i in trg:
            if len(i) == 1 and i not in self.count_table.alphabet:  # Bytes
                ids.extend(list(i.encode("utf-8")))
            else:
                ids.append(keys.get(i, unk_id) + shift)

        if unk_id < 0:
            ids = [i for i in ids if i >= 0]

        if eos_id >= 0:
            ids.append(eos_id)

        if pad_length is not None and pad_id >= 0:
            l = len(ids) - pad_length
            if l < 0:
                raise Exception("Sequence too long, invalid pad")
            ids = ids + [pad_id] * l

        return ids

    def decode(
        self,
        trg: list[int],
        unk: str = "<unk>",
        unk_id: int = 0,
        bos: str = "<s>",
        bos_id: int = 1,
        eos: str = "</s>",
        eos_id: int = 2,
        pad: str = "<pad>",
        pad_id: int = 3,
    ) -> list[str]:
        """
        Decode a sequence of ids into tokens.

        Args:
            trg: list of ids.
            unk: string representation of the UNK token.
            unk_id: id for the UNK token.
            bos: string representation of the BOS token.
            bos_id: id for the BOS token.
            eos: string representation of the EOS token.
            eos_id: id for the EOS token.
            pad: string representation of the pad token.
            pad_id: id for the pad token

        Returns:
            List of string tokens.
        """
        pieces = []
        tmp_bytes = []
        reserved = max(0, unk_id, bos_id, eos_id, pad_id)
        reserved_map = {unk_id: unk, bos_id: bos, eos_id: eos, pad_id: pad}
        shift = 256 + reserved

        for i in trg:
            if i < reserved:  # Special token
                pieces.append(reserved_map[i])
            elif i - reserved < 256:  # Bytes
                tmp_bytes.append(i - reserved)
            else:
                pieces.append(bytes(tmp_bytes).decode("utf8"))
                tmp_bytes = []
                pieces.append(self.count_table.get_span(i - shift))
        return pieces

    def export_vocab(
        self,
        unk: str = "<unk>",
        unk_id: int = 0,
        bos: str = "<s>",
        bos_id: int = 1,
        eos: str = "</s>",
        eos_id: int = 2,
        pad: str = "<pad>",
        pad_id: int = 3,
        to_yaml: bool = True,
    ) -> str | dict[str, int]:
        # reserved = max(-1, unk_id, bos_id, eos_id, pad_id)
        # reserved_map = {unk: unk_id, bos: bos_id, eos: eos_id, pad: pad_id}
        # reserved_map = {k: v for k, v in reserved_map.items() if v >= 0}
        # byte_map = {
        #     f"{WHITE_CHAR}{WHITE_CHAR}{i}": i + reserved + 1 for i in range(256)
        # }
        # shift = 256 + reserved + 1
        # span_map = self.count_table._span_index.todict()
        # span_map = {k: v + shift for k, v in span_map.items()}
        # vocab = {}
        # vocab.update(reserved_map)
        # vocab.update(byte_map)
        # vocab.update(span_map)
        # if to_yaml:
        #     vocab = yaml.dump(vocab, indent=2)
        # return vocab
        return self.export_vocab_static(
            self.count_table._span_index.todict(),
            unk,
            unk_id,
            bos,
            bos_id,
            eos,
            eos_id,
            pad,
            pad_id,
            to_yaml,
        )
    
    @staticmethod
    def export_vocab_static(
        span_map: dict[str, int], 
        unk: str = "<unk>",
        unk_id: int = 0,
        bos: str = "<s>",
        bos_id: int = 1,
        eos: str = "</s>",
        eos_id: int = 2,
        pad: str = "<pad>",
        pad_id: int = 3,
        to_yaml: bool = True,
    ) -> str | dict[str, int]:
        reserved = max(-1, unk_id, bos_id, eos_id, pad_id)
        reserved_map = {unk: unk_id, bos: bos_id, eos: eos_id, pad: pad_id}
        reserved_map = {k: v for k, v in reserved_map.items() if v >= 0}
        byte_map = {
            f"{WHITE_CHAR}{WHITE_CHAR}{i}": i + reserved + 1 for i in range(256)
        }
        shift = 256 + reserved + 1
        span_map = {k: v + shift for k, v in span_map.items()}
        vocab = {}
        vocab.update(reserved_map)
        vocab.update(byte_map)
        vocab.update(span_map)
        if to_yaml:
            vocab = yaml.dump(vocab, indent=2)
        return vocab

    def save(self, file: IO[bytes]) -> None:
        """
        Save the model in a tar.gz file.
        The file contains "table.pkl" with information about the cooccurency table,
        "table_data.npz" with the content of the table and "self_data.pkl" with information
        of this model.

        Args:
            file: writable file (it can be a `io.BytesIO` object).
        """
        # Save table
        with tarfile.open(fileobj=file, mode="w:gz") as tar:
            with io.BytesIO() as table, io.TextIOWrapper(io.BytesIO()) as self_data:
                self.count_table.save(table)

                # Save self
                data = {
                    "vocab_size": self.vocab_size,
                    "maxlen": self.maxlen,
                    "pretokenize": self.pretokenize,
                    "seed": self.seed,
                    "type": self.__class__.__name__,
                }
                json.dump(data, self_data, indent=2)

                info = tarfile.TarInfo("table.tar.gz")
                info.mtime = time.time()
                info.size = table.tell()
                table.seek(0)
                tar.addfile(info, table)

                info = tarfile.TarInfo("self_data.json")
                info.mtime = time.time()
                info.size = self_data.tell()
                self_data.seek(0)
                tar.addfile(info, self_data.buffer)

    @classmethod
    def load(cls, file: IO[bytes]) -> PairedSPModel:
        """
        Load a model from a tar.gz file.

        Args:
            file: readable file object (a tar.gz file).

        Returns:
            Model from the file.
        """
        with tarfile.open(fileobj=file, mode="r:gz") as tar:
            table = tar.extractfile("table.tar.gz")
            table = Table.load(table)
            with tar.extractfile("self_data.json") as buff:
                data = json.load(buff)
            new_ = cls(
                vocab_size=data["vocab_size"],
                maxlen=data["maxlen"],
                pretokenize=data["pretokenize"],
                seed=data["seed"],
            )
            new_.count_table = table
            new_.update_prob_table()
        return new_

    @classmethod
    def load_vocab(cls, file: IO[bytes]) -> dict[str, int]:
        """
        Load the vocabulary from a tar.gz file.

        Args:
            file: readable file object (a tar.gz file).

        Returns:
            Vocabulary as a dictionary {string: int}.
        """
        with tarfile.open(fileobj=file, mode="r:gz") as tar:
            table = tar.extractfile("table.tar.gz")
            vocab = Table.load_vocab(table)
            with tar.extractfile("self_data.json") as buff:
                data = json.load(buff)
            vocab_size=data["vocab_size"]
            if vocab_size != len(vocab):
                logger.warning(f"Mismatch vocabulary size: expected {vocab_size}, got {len(vocab)}")
        return vocab
