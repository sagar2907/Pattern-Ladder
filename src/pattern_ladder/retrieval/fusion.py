"""Reciprocal Rank Fusion.

BM25 scores and cosine similarities are not comparable: they have different
scales, different distributions, and the ratio between them varies per query.
Any attempt to combine them by weighted sum requires normalising scores, and
every normalisation (min-max, z-score) is itself query-dependent and unstable
when one arm returns a single result.

RRF sidesteps this by discarding the scores and using only ranks. That is a
real loss of information -- it cannot tell a runaway top hit from a marginal
one -- but it buys robustness that matters more here, where the two arms
disagree by design.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .. import config


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[tuple[int, float]]],
    *,
    k: int = config.RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists into one.

    Args:
        rankings: one iterable of (doc_id, score) per retriever, best first.
            Only the position is used; the score is ignored by design.
        k: the RRF damping constant. Larger k flattens the contribution of top
            ranks, making the fusion more democratic between arms.
        weights: optional per-retriever multipliers, same length as `rankings`.

    Returns:
        (doc_id, fused_score) sorted by descending score. Ties are broken by
        ascending doc_id so the output is deterministic -- without this, two
        documents with identical fused scores could swap between runs and make
        the smoke tests flaky.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(
            f"got {len(weights)} weights for {len(rankings)} rankings"
        )

    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (doc_id, _score) in enumerate(ranking):
            # rank is 0-based here; the canonical formula is 1-based, hence +1.
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
