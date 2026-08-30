"""Tests for graph construction and the embedding backfill."""

from __future__ import annotations

import numpy as np
import pytest

from pattern_ladder.graph.build import add_knn_backfill, build_link_graph, describe


def test_every_problem_becomes_a_node_even_with_no_links(problems):
    """Isolated problems must exist as nodes or the backfill cannot find them,
    and they would silently vanish from the corpus the graph describes."""
    graph = build_link_graph(problems)
    assert graph.number_of_nodes() == len(problems)
    assert graph.degree("lonely-math") == 0


def test_links_are_undirected_and_deduplicated(problems):
    graph = build_link_graph(problems)
    assert graph.has_edge("window-easy", "window-medium")
    assert graph.has_edge("window-medium", "window-easy")


def test_curated_edges_are_labelled_and_weighted(problems):
    graph = build_link_graph(problems)
    data = graph.edges["window-easy", "window-medium"]
    assert data["source"] == "link"
    assert data["weight"] == pytest.approx(1.0)


def test_backfill_attaches_isolated_nodes(problems, embeddings):
    graph = build_link_graph(problems)
    assert graph.degree("lonely-array") == 0
    add_knn_backfill(graph, problems, embeddings)
    assert graph.degree("lonely-array") > 0


def test_backfill_edges_are_labelled_and_weighted_lower(problems, embeddings):
    """Louvain optimises weighted modularity, so an inferred edge must count
    for less than a curated one or the families get redrawn by embeddings."""
    graph = build_link_graph(problems)
    add_knn_backfill(graph, problems, embeddings)
    inferred = [d for _u, _v, d in graph.edges(data=True) if d["source"] == "knn"]
    assert inferred
    assert all(d["weight"] < 1.0 for d in inferred)


def test_shared_tag_requirement_blocks_unrelated_neighbours(problems, embeddings):
    """Regression: similarity alone attached problems that share no topic.

    On the real corpus this put Valid Anagram in a shortest-path family. Here,
    lonely-math shares no tag with anything, so with the requirement on it
    stays isolated and without it it gets attached.
    """
    strict = build_link_graph(problems)
    add_knn_backfill(strict, problems, embeddings, min_similarity=0.0, require_shared_tag=True)
    assert strict.degree("lonely-math") == 0

    loose = build_link_graph(problems)
    add_knn_backfill(loose, problems, embeddings, min_similarity=0.0, require_shared_tag=False)
    assert loose.degree("lonely-math") > 0


def test_raising_the_similarity_floor_admits_fewer_edges(problems, embeddings):
    """The floor is the main lever on how much inferred structure is added, so
    it must actually bite. Asserting monotonicity rather than a fixed count
    keeps the test meaningful if the fixture vectors change."""
    counts = []
    for floor in (0.0, 0.9, 1.01):
        graph = build_link_graph(problems)
        add_knn_backfill(graph, problems, embeddings, min_similarity=floor)
        counts.append(describe(graph).knn_edges)
    assert counts[0] >= counts[1] >= counts[2]
    # A floor above the maximum possible cosine must admit nothing at all.
    assert counts[2] == 0


def test_well_connected_nodes_are_not_backfilled(problems, embeddings):
    """Backfilling nodes that already have curated links adds edges that bridge
    distinct families and merges them."""
    graph = build_link_graph(problems)
    before = graph.degree("window-medium")
    add_knn_backfill(graph, problems, embeddings, max_degree=0)
    assert graph.degree("window-medium") == before


def test_backfill_never_creates_self_loops(problems, embeddings):
    graph = build_link_graph(problems)
    add_knn_backfill(graph, problems, embeddings, min_similarity=0.0)
    assert not list(__import__("networkx").selfloop_edges(graph))


def test_mismatched_embedding_shape_raises(problems):
    with pytest.raises(ValueError):
        add_knn_backfill(build_link_graph(problems), problems, np.zeros((3, 3), dtype=np.float32))


def test_backfill_is_deterministic(problems, embeddings):
    first = build_link_graph(problems)
    add_knn_backfill(first, problems, embeddings)
    second = build_link_graph(problems)
    add_knn_backfill(second, problems, embeddings)
    assert sorted(first.edges()) == sorted(second.edges())


def test_describe_counts_edges_by_provenance(problems, embeddings):
    graph = build_link_graph(problems)
    link_only = describe(graph)
    assert link_only.knn_edges == 0
    add_knn_backfill(graph, problems, embeddings)
    after = describe(graph)
    assert after.link_edges == link_only.link_edges
    assert after.edges == after.link_edges + after.knn_edges
    assert after.isolated <= link_only.isolated
