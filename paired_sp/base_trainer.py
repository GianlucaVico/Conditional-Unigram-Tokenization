import ipdb
import numpy as np
from .dataloader import Dataloader
from .sentencepiece_spans import SentencePieceSpans
from .base_model import PairedSPModel
from .utils import INF
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PairedSPTrainer:
    def __init__(
        self,
        model: PairedSPModel,
        src_dl: Dataloader[list[str]],
        trg_dl: Dataloader[str],
        reduce: float = 0.4,
        n_iter: int = 1,
        n_sub_iter: int = 1,
        pretokenize: bool = False,
        char_coverage: float = 1.0,
        threshold: int = 0,
        spans: int | None = None,
        spans_runs: int = 3,
        temp_file: str | None = None,
    ) -> None:
        """
        Trainer for `PairedSPModel`.

        Args:
            models: model to train.
            src_dl: dataloader with tokenized sentences in the source language.
            trg_dl: dataloader with non-tokenized sentences in the target language.
            reduce: remove this portion of the target vocabulary at every iteration.
            em_iter: maximum number of iterations of the EM algorithm.
            em_sub_iter: reduce the vocabulary after this many EM iterations.
            pretokenize: split the target sentence on black spaces.
            char_coverage: keep this portion of charcters. The least frequents are removed from the vocabulary.
            threshold: remove spans that occour less than the threshold.
            spans: use a fixed number of spans. If None, use all the spans / other heuristics.
            spans_runs: perform multiple run to obtain the initial spans.
            dtype: data type of the cooccurency table.
            temp_file: path of the temporary files. If None, temp. file is not used.
        """
        self.model = model
        # Parameters
        self.reduce = reduce
        self.n_iter = n_iter
        self.n_sub_iter = n_sub_iter
        self.pretokenize = pretokenize
        self.char_coverage = char_coverage
        self.threshold = threshold
        self.spans = spans
        self.spans_runs = spans_runs

        # Files
        self.src_dl = src_dl
        self.trg_dl = trg_dl
        self.temp_file = temp_file

    def prune(self, *, quiet: bool = True) -> bool:
        """
        Remove the least frequent spans from the table and the dictionary.

        Returns:
            True if the model reached the desired vocabulary size; False otherwise.
        """
        # ipdb.set_trace()
        count_table = self.model.count_table
        logger.debug("Pruning")
        count_table.generate_inv_indexes()

        vocab_size = self.model.vocab_size
        curr_size = count_table.n_spans

        counts = []
        for span, span_index in tqdm(
            count_table._span_index, "Prune", disable=quiet, leave=False
        ):
            if len(span) > 1:
                cst = count_table.get_dict_by_id(span_index)  # {tok_id: count}
                cs = count_table.count_span[span_index]
                if cs == 0:
                    logger.warning(f"{span} ({span_index}) disappeared.")
                    counts.append((-INF, span))
                info = 0
                for tok_id, count in cst.items():
                    ct = count_table.count_tok[tok_id]
                    if count != 0 and ct != 0:                         
                        info += count * (
                            np.log(count)
                            - np.log(ct)
                            - np.log(cs)
                            + count_table.lognorm
                        )
                counts.append((info / count_table.norm, span))

        # Find portion to delete
        to_delete = min(int(self.reduce * curr_size), curr_size - vocab_size)

        # Delete
        counts = sorted(counts)
        del_list = [i[1] for i in counts[:to_delete]]
        count_table.delete_spans(del_list)

        count_table.update_prob_table()
        logger.info(f"New size: {curr_size - len(del_list)}/{vocab_size}")

        # Fix max length
        if len(del_list) != 0:
            max_del_length = max([len(i) for i in del_list])
            if max_del_length >= self.model.maxlen:
                self.model.maxlen = max(map(len, self.model.count_table.spans))
        # True: done, False: more iterations
        count_table.optimize()
        return count_table.n_spans <= vocab_size

    def prune_character(self) -> None:
        """
        Remove the least frequent characters in the alphabet.
        `char_coverage` defines how much "mass" of characters is kept.

        E.g. 0.95 coverage removes the characters whose frequency is th 5% of all chacters
        and not the 5% of low-frequency characters. If the original alphabet had 100 characters,
        this does not mean it will be reduced to 95 characters.
        """
        logger.debug("Pruning characters")
        count_table = self.model.count_table
        count_table.generate_inv_indexes()

        counts = count_table.sum(axis=1)  # {id: count}
        counts = {
            count_table.get_span(k): v for k, v in counts.items()
        }  # {span: count}
        counts = {k: v for k, v in counts.items() if k in count_table.alphabet}
        counts = list(counts.items())  # [(span, count), ...]
        counts = sorted(counts, key=lambda x: x[1])

        total = sum([i[1] for i in counts])
        remove_count = (1 - self.char_coverage) * total

        removed = 0
        char_del_list = []
        for char, count in counts:
            if removed + count > remove_count:
                break
            char_del_list.append(char)
            removed += count

        span_del_list = []
        for span in count_table.spans:
            if any(i in span for i in char_del_list):
                span_del_list.append(span)
        # Chars to delete
        count_table.delete_spans(char_del_list)
        count_table.delete_spans(span_del_list)

        del count_table.alphabet  # Force clear cache
        count_table.lock_alphabet = True

    def fit(self, *, quiet: bool = True) -> None:
        """
        Train the model with the data in the source and target dataloaders.

        Args:
            quiet: disable tqdm.
        """
        # Init
        logger.debug("Init")
        span_list = []
        if self.spans is not None and self.spans > 0:
            logger.debug("Sentepiece list")
            span_list = SentencePieceSpans(self.spans)
            span_list.train(self.trg_dl, self.spans_runs)
            logger.info(f"SentencePiece spans: {len(span_list)}")

        logger.debug("Init table")
        self.model.init_table(
            self.src_dl,
            self.trg_dl,
            self.pretokenize,
            self.threshold,
            span_list,
            quiet=quiet,
        )
        logger.info(f"Spans: {self.model.count_table.n_spans}")
        logger.info(f"Tokens: {self.model.count_table.n_tokens}")
        logger.info(f"Table: {self.model.count_table.size}")

        if self.char_coverage < 1:
            self.prune_character()

        n_spans = self.model.count_table.n_spans
        if self.model.vocab_size > n_spans:
            logger.warning(
                f"Vocab size is larger than the availble number of tokens {n_spans}."
            )
            logger.warning(f"Set vocab size to {n_spans}")
            self.model.vocab_size = n_spans

        logger.info(f"Alphabet size: {len(self.model.count_table.alphabet)}")

        logger.info("Star training")

        self._fit(quiet=quiet)

        logger.info("Training complete")
        self.model.vocab_size = self.model.count_table.n_spans
        self.model.count_table.generate_inv_indexes()

        logger.info(f"Spans: {self.model.count_table.n_spans}")
        logger.info(f"Tokens: {self.model.count_table.n_tokens}")
        logger.info(f"Table: {self.model.count_table.size}")

    def _fit(self, *, quiet: bool = True) -> None:
        raise NotImplementedError("Subclasses have to define the training loop.")
