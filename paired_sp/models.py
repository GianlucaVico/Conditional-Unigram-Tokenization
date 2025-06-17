"""
Different variants of the PairedSP model.

Each subclass defines a different scoring function and alignment method.
"""

import functools
import ipdb
import logging
from .base_model import PairedSPModel
import numpy as np


from .utils import TokenMask

EXTRACT_N = 10

# TODO fix align, fix extract

logger = logging.getLogger(__name__)

try:
    from scipy.special import digamma
except ImportError:
    logger.warning("Scipy not found")
    digamma = np.log


class PairedSPMaxModel(PairedSPModel):
    """Use the maximum join probability to score and align spans with tokens."""

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Score.
        """
        max_ = max(self.count_table.get_by_id(span_id, src_ids))
        return digamma(max_) - self.count_table.lognorm

    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:

        # Query table
        token_mask = self.count_table.get_token_mask(src)

        scores = []
        toks = []
        for span in trg:
            span_id = self.count_table.get_span_id(span)
            if span_id is not None:
                counts = self.count_table.get_by_id(span_id, token_mask)
                score = max(counts)
                tok = counts.index(score)
                scores.append(score)
                toks.append(tok)
        # span index is implicit
        return list(zip(toks, range(len(trg)), scores))

    def extract_pairs(self, k: int | None = 1) -> list[tuple[str, str, float]]:
        """
        Find the corresponding spans and tokens and score them.

        Args:
            k: top-k parameter. If None, use the entire table

        Returns:
            List of (span, token, score)
        """
        if k is None:
            k = self.count_table.n_tokens
        spans = self.count_table.n_spans
        index = 0
        pairs = np.zeros((spans * k, 2), np.int_)
        scores = np.zeros(spans * k)
        while index < spans:
            next_index = index + EXTRACT_N
            rows = self.count_table.table[index:next_index]
            if rows.size != 0:
                rows = rows.todense()
                part_tokens = np.argpartition(rows, k, axis=1)[:, -k:]
                part_scores = np.take_along_axis(rows, part_tokens, 1)

                pairs[index:next_index, 0] = np.arange(index, next_index).repeat(k)
                pairs[index:next_index, 1] = part_tokens.ravel()
                scores[index:next_index] = part_scores.ravel()
            index = next_index
        return map(
            lambda x: (
                self.count_table.id_span(x[0][0]),
                self.count_table.id_token(x[0][1]),
                x[1],
            ),
            zip(pairs, scores),
        )


class PairedSPSumModel(PairedSPMaxModel):
    """
    Marginalize the join probability to score the spans and tokens.

    Use p(s, T).
    """

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Score.
        """
        sum_ = sum(self.count_table.get_by_id(span_id, src_ids))
        return digamma(sum_) - self.count_table.lognorm

    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:

        # Query table
        token_mask = self.count_table.get_token_mask(src)

        alignments = []
        for span_index, span in enumerate(trg):
            span_id = self.count_table.get_span_id(span)
            if span_id is not None:
                counts = self.count_table.get_by_id(span_id, token_mask)
                for tok_index, score in enumerate(counts):
                    if score != 0:
                        alignments.append((tok_index, span_index, score))
        return alignments


class UnpairedSPMaxModel(PairedSPModel):
    """Score the spans without accounting for the source sentence."""

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Score.
        """
        max_ = max(self.count_table.get_by_id(span_id))
        return digamma(max_) - self.count_table.lognorm


class UnpairedSPSumModel(PairedSPModel):
    """Score the spans without accounting for the source sentence."""

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Score.
        """
        # log sum_t p(s, t)
        sum_ = sum(self.count_table.get_by_id(span_id))
        return digamma(sum_) - self.count_table.lognorm


class PairedSPExactCondModel(PairedSPModel):
    """Score the spans and tokens with the conditional probability of span given a token."""

    @np.errstate(divide="ignore")
    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:
        # src - trg - score
        token_mask = self.count_table.get_token_mask(src)

        ct = self.count_table.get_token_count(token_mask)

        alignments = []
        for span_index, span in enumerate(trg):
            span_id = self.count_table.get_span_id(span)
            if span_id is not None:
                cst = self.count_table.get_by_id(span_id, token_mask)
                pst = [
                    i / j for i, j in zip(cst, ct)
                ]  # p(s|t) for each t in the sentence
                score = max(pst)  # highest p(s|t)
                tok_index = pst.index(score)  # index of t
                alignments.append((tok_index, span_index, score))
        return alignments

    @np.errstate(divide="ignore")
    def extract_pairs(self, k: int | None = 1) -> list[tuple[str, str, float]]:
        """
        Find the corresponding spans and tokens and score them.

        Args:
            k: top-k parameter. If None, use the entire table

        Returns:
            List of (span, token, score)
        """
        if k is None:
            k = self.count_table.n_tokens
        spans = self.count_table.n_spans
        index = 0
        pairs = np.zeros((spans * k, 2), np.int_)
        scores = np.zeros(spans * k)
        while index < spans:
            next_index = min(index + EXTRACT_N, spans)
            rows = self.count_table.table[index:next_index]
            if rows.size != 0:
                rows = rows.todense()
                rows = rows / self._count_t
                # rows = np.log(rows) - np.log(self._count_t)

                part_tokens = np.argpartition(rows, -k, axis=1)[:, -k:]
                part_scores = np.take_along_axis(rows, part_tokens, 1)

                pairs[index * k : next_index * k, 0] = np.arange(
                    index, next_index
                ).repeat(k)
                pairs[index * k : next_index * k, 1] = part_tokens.ravel()
                scores[index * k : next_index * k] = part_scores.ravel()
            index = next_index
        return map(
            lambda x: (
                self.count_table.id_span(x[0][0]),
                self.count_table.id_token(x[0][1]),
                x[1],
            ),
            zip(pairs, scores),
        )

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        """
        Score a span given a sentence in the source language.

        Args:
            src_ids: ids of the source tokens.
            span_id: id of the target span.

        Returns:
            Approximation of Log p(s | T).
        """
        cst = self.count_table.get_by_id(
            span_id, src_ids
        )  # Count(s=span, t | T) Row 1 x len(src)
        ct = self.count_table.get_token_count(src_ids)  # Count(t | T) Row 1 x len(src)        
        return np.log(sum(cst)) - np.log(sum(ct))   # TODO ct or cs
        # return digamma(sum(cst)) - digamma(sum(ct))


class PairedSPPMI(PairedSPModel):
    """
    Score spans and tokens based on their mutual information.
    The idea is that aligned spans and tokens are not independent.

    Independent spans and tokens have 0 MI.
    """

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        # Non conditional version: pmi(s, t) = log_2 count(s, t) - log_2 count(s) - log_2 count(t) + log_2 norm
        cst = self.count_table.get_by_id(span_id, src_ids)  # count(s, t), row
        cs = self.count_table.count_span[span_id]  # c(s), single
        ct = self.count_table.get_token_count(src_ids)
        pmi = np.log(cst) - np.log(cs) - np.log(ct) + np.log(self.count_table.norm)
        return max(pmi)

    @np.errstate(divide="ignore")
    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:
        # src - trg - score
        token_mask = self.count_table.get_token_mask(src)
        ct = self.count_table.get_token_count(token_mask)

        alignments = []
        for span_index, span in enumerate(trg):
            span_id = self.count_table.get_span_id(span)
            if span_id is not None:
                cst = self.count_table.get_by_id(span_id, token_mask)
                cs = self.count_table.count_span[span_id]
                # pmi = np.log(cst) - np.log(cs) - np.log(ct) + np.log(self.norm)
                pmi = [
                    np.log(i) - np.log(cs) - np.log(j) + self.count_table.lognorm
                    for i, j in zip(cst, ct)
                ]
                score = max(pmi)  # highest p(s|t)
                tok_index = pmi.index(score)  # index of t
                alignments.append((tok_index, span_index, score))
        return alignments

    def extract_pairs(self, k: int | None = 1) -> list[tuple[str, str, float]]:
        """
        Find the corresponding spans and tokens and score them.

        Args:
            k: top-k parameter. If None, use the entire table

        Returns:
            List of (span, token, score)
        """
        if k is None:
            k = self.count_table.n_tokens
        spans = self.count_table.n_spans
        index = 0
        pairs = np.zeros((spans * k, 2), np.int_)
        scores = np.zeros(spans * k)
        while index < spans:
            next_index = index + EXTRACT_N
            rows = self.count_table.table[index:next_index]
            if rows.size != 0:
                rows = rows.todense()
                cs = self._count_s[index:next_index].reshape(-1, 1)
                ct = self._count_t.reshape(1, -1)
                rows = np.log(rows) - np.log(cs) - np.log(ct) + np.log(self.norm)

                part_tokens = np.argpartition(rows, k, axis=1)[:, -k:]
                part_scores = np.take_along_axis(rows, part_tokens, 1)

                pairs[index:next_index, 0] = np.arange(index, next_index).repeat(k)
                pairs[index:next_index, 1] = part_tokens.ravel()
                scores[index:next_index] = part_scores.ravel()
            index = next_index
        return map(
            lambda x: (
                self.count_table.id_span(x[0][0]),
                self.count_table.id_token(x[0][1]),
                x[1],
            ),
            zip(pairs, scores),
        )


class PairedSPCPMI(PairedSPPMI):
    """
    Score spans and tokens based on their mutual information.
    The idea is that aligned spans and tokens are not independent.

    Independent spans and tokens have 0 MI.
    """

    def score_pair(self, src_ids: TokenMask, span_id: int) -> float:
        # Non conditional version: pmi(s, t) = log_2 count(s, t) - log_2 count(s) - log_2 count(t) + log_2 norm
        src_ids = np.unique(src_ids)
        cst = self.count_table.get_by_id(span_id, src_ids)  # count(s, t), row
        ct = self.count_table.get_token_count(src_ids)  # c(t), row

        # c(s, t) | c(s) = sum t in T c(s,t) | c(t) | tot = sum s in V t in T c(s, t)
        pmi = np.log(cst) - np.log(sum(cst)) - np.log(ct) + np.log(sum(ct))
        return max(pmi)

    @np.errstate(divide="ignore")
    def align(self, src: list[str], trg: list[str]) -> list[tuple[int, int, float]]:
        # src - trg - score
        token_mask = self.count_table.get_token_mask(src)
        ct = self.count_table.get_token_count(token_mask)
        lognorm = np.log(sum(ct))
        alignments = []
        for span_index, span in enumerate(trg):
            span_id = self.count_table.get_span_id(span)
            if span_id is not None:
                cst = self.count_table.get_by_id(span_id, token_mask)
                cs = sum(cst)
                # pmi = np.log(cst) - np.log(sum_t cst) - np.log(ct) + np.log(self.norm)
                cpmi = [
                    np.log(i) - np.log(cs) - np.log(j) + lognorm
                    for i, j in zip(cst, ct)
                ]
                score = max(cpmi)  # highest cpmi(s|t)
                tok_index = cpmi.index(score)  # index of t
                alignments.append((tok_index, span_index, score))
        return alignments
