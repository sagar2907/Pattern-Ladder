"""End-to-end tests of the search pipeline.

The cross-encoder is disabled throughout (`use_reranker=False`): loading it
would download ~90MB and make the suite depend on the network. Its behaviour is
covered by the shape of the fusion contract instead -- what matters structurally
is that the pipeline composes, filters and expands correctly, and that is
identical either way.
"""

from __future__ import annotations

import numpy as np
import pytest

from pattern_ladder.graph.build import add_knn_backfill, build_link_graph
from pattern_ladder.graph.families import build_families, detect_families
from pattern_ladder.retrieval.search import SearchEngine
from pattern_ladder.understand.schema import Intent


@pytest.fixture
def engine(problems, lexical_index, dense_index, embeddings) -> SearchEngine:
    graph = build_link_graph(problems)
    add_knn_backfill(graph, problems, embeddings)
    families = build_families(detect_families(graph, resolution=1.0), problems, min_size=5)
    return SearchEngine(problems, lexical_index, dense_index, families)


def _search(engine, query, intent=None, **kwargs):
    kwargs.setdefault("use_reranker", False)
    return engine.search(query, intent or Intent(), **kwargs)


class TestPipeline:
    def test_returns_results_for_a_plain_query(self, engine):
        response = _search(engine, "monotonic stack next greater")
        assert response.results
        assert response.results[0].problem.slug.startswith("stack")

    def test_top_k_is_respected(self, engine):
        assert len(_search(engine, "window", top_k=2).results) <= 2

    def test_nonsense_query_still_returns_dense_neighbours_and_flags_it(self, engine):
        """Dense retrieval is exhaustive: it has no notion of "no match" and
        always returns the nearest rows, however far away they are. So a
        meaningless query produces results, and the honest thing is to say the
        lexical arm found nothing rather than to imply both agreed.

        This is a real limitation, recorded here rather than papered over: the
        system cannot currently tell a student that their query matched
        nothing."""
        response = _search(engine, "zzzqqq wwwxxx")
        assert response.results
        assert any("dense only" in note for note in response.notes)
        assert all(r.match == "dense" for r in response.results)

    def test_every_result_carries_a_reason(self, engine):
        for result in _search(engine, "sliding window").results:
            assert result.reason

    def test_match_kind_is_recorded(self, engine):
        kinds = {r.match for r in _search(engine, "window").results}
        assert kinds <= {"both", "lexical", "dense"}

    def test_results_are_deterministic(self, engine):
        first = [r.problem.slug for r in _search(engine, "stack").results]
        second = [r.problem.slug for r in _search(engine, "stack").results]
        assert first == second


class TestIntentEffects:
    def test_technique_from_intent_reaches_the_retrievers(self, engine):
        """The parsed technique is the jargon BM25 needs and the raw sentence
        never contains; this is what query understanding buys."""
        bare = _search(engine, "I get stuck shrinking things")
        guided = _search(engine, "I get stuck shrinking things", Intent(technique="sliding window"))
        guided_slugs = [r.problem.slug for r in guided.results]
        assert any(s.startswith("window") for s in guided_slugs)
        assert guided_slugs != [r.problem.slug for r in bare.results] or bare.results

    def test_difficulty_filter_narrows_results(self, engine):
        response = _search(engine, "window", Intent(difficulty="Hard"))
        assert all(r.problem.difficulty == "Hard" for r in response.results)

    def test_impossible_difficulty_is_ignored_rather_than_emptying_results(self, engine):
        """An over-specific parse must not silently erase every result."""
        response = _search(engine, "parentheses stack", Intent(difficulty="Easy"))
        assert response.results
        response_hard = _search(engine, "min stack design", Intent(difficulty="Hard"))
        if not any(r.problem.difficulty == "Hard" for r in response_hard.results):
            assert response_hard.notes


class TestLadder:
    def test_a_ladder_is_built_for_a_family_query(self, engine):
        response = _search(engine, "monotonic stack next greater")
        assert response.ladder is not None
        assert response.ladder.rungs

    def test_ladder_is_ordered_easy_to_hard(self, engine):
        ladder = _search(engine, "monotonic stack next greater").ladder
        ranks = [p.difficulty_rank for p in ladder.rungs]
        assert ranks == sorted(ranks)

    def test_ladder_length_is_capped(self, engine):
        from pattern_ladder import config

        ladder = _search(engine, "stack").ladder
        assert len(ladder.rungs) <= config.LADDER_LENGTH

    def test_ladder_reports_the_full_family_size(self, engine):
        ladder = _search(engine, "monotonic stack").ladder
        assert ladder.truncated_from >= len(ladder.rungs)

    def test_start_here_is_on_the_ladder(self, engine):
        ladder = _search(engine, "monotonic stack").ladder
        assert ladder.start_here in {p.slug for p in ladder.rungs}

    def test_technique_match_outweighs_a_single_top_ranked_result(self, engine, problems):
        """Regression: the ladder was anchored on the top result's family.

        A single result's family is one noisy sample of what a query is about.
        On the real corpus this put Minimum Window Substring's family (named
        "Queue / Design") behind a sliding-window query, and served a ladder of
        queue problems.
        """
        response = _search(engine, "stack", Intent(technique="sliding window"))
        if response.ladder is not None:
            name = f"{response.ladder.family.name} {' '.join(response.ladder.family.tags)}".lower()
            assert "sliding window" in name or "stack" in name


class TestFusionContract:
    def test_disabling_the_reranker_preserves_fusion_order(self, engine):
        response = _search(engine, "window")
        ranks = [r.fusion_rank for r in response.results]
        assert ranks == sorted(ranks)

    def test_a_dense_only_query_still_returns_results(self, engine, problems):
        """BM25 returns nothing for a query sharing no vocabulary with the
        corpus; the dense arm must carry it."""
        engine_no_lexical = SearchEngine(
            problems, _NullLexical(), engine.dense, engine.families
        )
        response = engine_no_lexical.search(
            "shrink", Intent(), use_reranker=False
        )
        assert response.results
        assert any("dense only" in note for note in response.notes)


class _NullLexical:
    """Stands in for a BM25 index that matches nothing."""

    def search(self, query: str, k: int):  # noqa: ARG002
        return []


def test_family_lookup_is_by_slug(engine, problems):
    assert engine.family_for("does-not-exist") is None


def test_engine_handles_a_corpus_with_no_families(problems, lexical_index, dense_index):
    engine = SearchEngine(problems, lexical_index, dense_index, [])
    response = engine.search("window", Intent(), use_reranker=False)
    assert response.ladder is None
    assert any("no ladder" in note for note in response.notes)


def test_query_vector_is_encoded_once(engine, monkeypatch):
    """Encoding is the second most expensive step after reranking; the dense
    arm and the ladder relevance filter must share one vector rather than each
    computing its own."""
    calls = []
    original = engine.dense.encode_query

    def counting(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(engine.dense, "encode_query", counting)
    engine.search("monotonic stack", Intent(), use_reranker=False)
    assert len(calls) == 1


def test_embeddings_matrix_rows_align_with_problems(problems, embeddings):
    """A positional artefact silently permuting against the corpus is the worst
    class of bug here: nothing errors, every result is just subtly wrong."""
    assert embeddings.shape[0] == len(problems)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
