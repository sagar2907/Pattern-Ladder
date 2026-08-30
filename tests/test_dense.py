"""Tests for the dense index.

These use a hand-built matrix rather than a real encoder, so they run with no
model download and can assert exact neighbour relationships.
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_ladder.index.dense import DenseIndex


def test_search_vector_ranks_by_dot_product(dense_index, problems):
    window_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hits = dense_index.search_vector(window_axis, k=3)
    assert all(problems[doc].slug.startswith(("window", "lonely-array")) for doc, _ in hits)


def test_results_are_ordered_by_descending_similarity(dense_index):
    hits = dense_index.search_vector(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=6)
    scores = [s for _d, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_is_clamped_to_corpus_size(dense_index, problems):
    hits = dense_index.search_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=999)
    assert len(hits) == len(problems)


def test_zero_k_returns_nothing(dense_index):
    assert dense_index.search_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=0) == []


def test_rejects_non_matrix_input():
    with pytest.raises(ValueError):
        DenseIndex(np.zeros(5, dtype=np.float32))


def test_save_and_load_round_trip(dense_index, tmp_path):
    path = tmp_path / "emb.npy"
    dense_index.save(path)
    reloaded = DenseIndex.load(path)
    assert np.array_equal(reloaded.matrix, dense_index.matrix)


def test_matrix_is_float32_and_contiguous(dense_index):
    """Both are required for the single-BLAS-call search path to stay fast."""
    assert dense_index.matrix.dtype == np.float32
    assert dense_index.matrix.flags["C_CONTIGUOUS"]


def test_empty_index_returns_nothing():
    empty = DenseIndex(np.zeros((0, 3), dtype=np.float32))
    assert empty.search_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5) == []
