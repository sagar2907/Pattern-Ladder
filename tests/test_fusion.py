"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from pattern_ladder.retrieval.fusion import reciprocal_rank_fusion


def test_matches_the_closed_form():
    """1/(k+rank) summed over retrievers, with rank 1-based."""
    a = [(1, 9.0), (2, 8.0)]
    b = [(2, 0.9), (1, 0.8)]
    fused = dict(reciprocal_rank_fusion([a, b], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)


def test_scores_are_ignored_only_ranks_matter():
    """The whole point of RRF: BM25 scores and cosine similarities are not
    comparable, so magnitudes must not leak into the result."""
    a = [(1, 1000.0), (2, 0.001)]
    b = [(1, 0.5), (2, 0.4)]
    rescaled_a = [(1, 0.2), (2, 0.1)]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([rescaled_a, b])


def test_a_document_in_both_lists_beats_one_in_either():
    a = [(1, 1.0), (3, 0.5)]
    b = [(2, 1.0), (1, 0.5)]
    order = [doc for doc, _ in reciprocal_rank_fusion([a, b])]
    assert order[0] == 1


def test_ties_break_on_document_id_for_determinism():
    """Without a deterministic tie-break, equally scored documents could swap
    between runs and make the smoke evaluation flaky."""
    a = [(5, 1.0)]
    b = [(2, 1.0)]
    assert [doc for doc, _ in reciprocal_rank_fusion([a, b])] == [2, 5]


def test_empty_rankings_are_handled():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [d for d, _ in reciprocal_rank_fusion([[(1, 1.0)], []])] == [1]


def test_weights_shift_the_balance():
    a = [(1, 1.0), (2, 1.0)]
    b = [(2, 1.0), (1, 1.0)]
    weighted = dict(reciprocal_rank_fusion([a, b], weights=[3.0, 1.0]))
    assert weighted[1] > weighted[2]


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[(1, 1.0)]], weights=[1.0, 2.0])


def test_non_positive_k_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[(1, 1.0)]], k=0)


def test_larger_k_flattens_the_advantage_of_rank_one():
    a = [(1, 1.0), (2, 1.0)]
    small = dict(reciprocal_rank_fusion([a], k=1))
    large = dict(reciprocal_rank_fusion([a], k=1000))
    assert small[1] / small[2] > large[1] / large[2]
