import functools
import concurrent.futures
import multiprocessing
import ipdb
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from . import (
    Dataloader,
    PairedSPTrainer,
    prepare_src,
    WHITE_CHAR,
    iter_spans_index,
    Table,
    PairedSPModel,
)
import logging

logger = logging.getLogger()

try:
    from scipy.special import softmax
    from scipy.linalg.lapack import dtrtri
except ImportError:
    logger.warning("Scipy not found")

    def softmax(
        x: npt.NDArray[np.float64], axis: int | None = None
    ) -> npt.NDArray[np.float64]:
        x_max = np.amax(x, axis=axis, keepdims=True)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

    def dtrtri(
        a: npt.NDArray[np.float64], lower: bool = False, overwrite_c: bool = False
    ) -> npt.NDArray[np.float64]:
        return np.linalg.inv(a), 0


class EMTrainer(PairedSPTrainer):
    def _fit(self, *, quiet=True) -> None:
        tokenizations, loss = self.compute_tokenizations(
            self.src_dl, self.trg_dl, quiet=quiet
        )
        logger.info(f"Initial loss={loss:.2f}")
        for i in (
            progress := tqdm(
                range(1, self.n_iter * self.n_sub_iter + 1),
                desc="Iteration",
                disable=quiet,
            )
        ):
            tokenizations, loss = self.paired_step(tokenizations, quiet=quiet)
            progress.set_postfix(
                {"Vocab. size": self.model.count_table.n_spans, "Loss": f"{loss:.3f}"}
            )
            # Prune after em_sub_iter
            if i % self.n_sub_iter == 0 and self.prune():
                break
        logger.info("Training loop completed")
        logger.info(f"Vocabulary size: {self.model.count_table.n_spans}")
        tokenizations, loss = self.compute_tokenizations(
            self.src_dl, self.trg_dl, quiet=quiet
        )
        self.update_tables(self.src_dl, tokenizations, quiet=quiet)

    def update_tables(
        self,
        src_texts: Dataloader[list[str]],
        trg_tokens: Dataloader[list[str]],
        *,
        quiet: bool = True,
    ) -> None:
        """
        Expectation step of the EM algorithms.
        Increase the counts in the cooccurency table.

        Args:
            src_text: dataloader with tokenized in the source language (list of tokens).
            trg_tokens: list of tokens in the target language (list of tokens).
            quiet: disable tqdm
        """
        logger.debug("Update tables")
        self.model.count_table.cache_clear()
        delta = self.model.count_table.get_linked_empty()
        # Expectation step
        for src_line, trg_spans in tqdm(
            zip(src_texts, trg_tokens), desc="Update table", disable=quiet, leave=False
        ):
            token_mask = self.model.count_table.get_token_mask(src_line)
            for span in trg_spans:
                span_id = self.model.count_table.get_span_id(span)
                if span_id is not None:
                    delta.increase_all_from_id(span_id, token_mask)

        self.model.count_table += delta
        # self.model.count_table += 1
        # self.model.count_table //= 2
        self.model.update_prob_table()

    def compute_tokenizations(
        self,
        src_texts: Dataloader[list[str]],
        trg_texts: Dataloader[str],
        *,
        quiet: bool = True,
    ) -> tuple[Dataloader[list[str]], float]:
        """
        Maximization step of the EM algorithm.
        The method compute the new optimal tokenization of the target sentences.

        Args:
            src_texts: dataloader with sentences in the source language (tokenized str).
            trg_texts: dataloader with sentences in the target language (tokenized str).
            quiet: disable tqdm.

        Returns:
            List of tokenized target sentences (list of tokens) an d the average loss.
        """
        # Maximization step
        logger.debug("Compute tokenizations")
        tot_loss = 0
        n = 0
        if self.temp_file is None:  # In memory
            tokenizations = []
            for src, trg in tqdm(
                zip(src_texts, trg_texts),
                desc="Comp. tokenization",
                disable=quiet,
                leave=False,
            ):
                loss, sizes = self.model.paired_forward(src, trg)
                tokenizations.append(self.model.paired_backward(trg, sizes))
                tot_loss += loss
                n += 1
            tokenizations = Dataloader(items=tokenizations, preload=True)
        else:  # Write to file
            with open(self.temp_file, "w") as temp:
                for src, trg in tqdm(
                    zip(src_texts, trg_texts),
                    desc="Comp. tokenization",
                    disable=quiet,
                    leave=False,
                ):
                    loss, sizes = self.model.paired_forward(src, trg)
                    temp.write(" ".join(self.model.paired_backward(trg, sizes)) + "\n")
                    tot_loss += loss
                    n += 1
            tokenizations = Dataloader(
                file=self.temp_file, fn=prepare_src, preload=False
            )
        return tokenizations, tot_loss / n

    def paired_step(
        self, tokenizations: Dataloader[list[str]], *, quiet: bool = True
    ) -> tuple[list[list[str]], float]:
        """
        Run one iteration of the EM algorithm.
        This method update the cooccurency table with the previous tokenization and
        then it coumputes the new tokenization.

        Args:
            tokenizations: list of tokenized sentences in the target language (list of tokens).
            quiet: disable tqdm.

        Returns:
            List of new tokenizations (list of tokens) and average loss.
        """
        self.update_tables(self.src_dl, tokenizations, quiet=quiet)
        tokenizations, loss = self.compute_tokenizations(
            self.src_dl, self.trg_dl, quiet=quiet
        )
        return tokenizations, loss


class MCTrainer(PairedSPTrainer):

    def _fit(self, *, quiet=True) -> None:
        compute_expected_counts = functools.partial(
            self.compute_expected_counts,
            model=self.model,
            src_dl=self.src_dl,
            trg_dl=self.trg_dl,
            pretokenize=self.pretokenize,
        )
        for i in (
            progress := tqdm(
                range(1, self.n_iter * self.n_sub_iter + 1),
                desc="Iteration",
                disable=quiet,
            )
        ):
            expected_table = compute_expected_counts(quiet=quiet)
            progress.set_postfix(
                {
                    "Vocab. size": self.model.count_table.n_spans,
                }
            )
            min_ =expected_table._count.min()
            max_ =expected_table._count.max()
            logger.info(f"Min count {min_}")
            logger.info(f"Max count {max_}")
            
            self.model.count_table.replace(expected_table)
            self.model.update_prob_table()
            # Prune after em_sub_iter
            if i % self.n_sub_iter == 0 and self.prune(quiet=quiet):
                break

        self._expected_table = self.model.count_table.get_linked_empty()
        expected_table = compute_expected_counts(quiet=quiet)
        self.model.count_table.replace(expected_table)
        self.model.update_prob_table()
        logger.info("Training loop completed")
        logger.info(f"Vocabulary size: {self.model.count_table.n_spans}")

    @staticmethod
    def compute_expected_counts(
        model: PairedSPModel | str,
        src_dl: Dataloader[list[str]],
        trg_dl: Dataloader[str],
        pretokenize: bool,
        *,
        quiet=True,
        model_type: type[PairedSPModel] = PairedSPModel,
    ) -> Table:
        if isinstance(model, str):
            path = model
            with open(path, "rb") as f:
                model = model_type.load(f)
        maxlen = model.maxlen
        count_table = model.count_table
        expected_table = model.count_table.get_linked_empty()
        for src, trg in tqdm(
            zip(src_dl, trg_dl),
            desc="Example",
            disable=quiet,
            leave=False,
            total=len(src_dl),
        ):
            token_mask = count_table.get_token_mask(src)
            srcl = len(token_mask)

            if pretokenize:
                trg_seqs = trg.split(WHITE_CHAR)
                trg_seqs = [i for i in trg_seqs if len(i) != 0]
                trg_seqs = [f"{WHITE_CHAR}{i}" for i in trg_seqs]
            else:
                trg_seqs = trg
            for trg in trg_seqs:

                trgl = len(trg)
                
                valid = []  # valid transitions start->end

                Tf = [
                    [-np.inf for _ in range(trgl + 2)] for _ in range(trgl + 2)
                ]  # Log-space, unnormalized transition probabilities,

                for i, j, span in iter_spans_index(trg, maxlen, pretokenize=pretokenize):
                    span_id = count_table.get_span_id(span)
                    if span_id is not None:
                        score = model.score_pair(token_mask, span_id) # TODO missing span                        
                        valid.append((i, j, span_id))
                        Tf[j][i] = score
                    elif j - i == 1:  # UNK char
                        Tf[j][i] = 0

                Tf = np.array(Tf).T
                # ipdb.set_trace()
                # Tf[-2, -1] = 0
                # Tf[-1, -1] = 0
                # Tf = softmax(Tf, axis=1)
                Tf = softmax(Tf)
                Qf = Tf[:-1, :-1]
                I = np.eye(len(Qf), order="F", dtype=np.float64)
                Nf, _ = dtrtri(I - Qf, lower=False, overwrite_c=True)

                betas = Nf[:, -1] # all 1s
                alphas = Nf[0]
                for j, i, span_id in valid:
                    expected_table.increase_all_from_id(
                        span_id, token_mask, Tf[j, i] * alphas[j] * betas[i] / srcl
                    )
        return expected_table

