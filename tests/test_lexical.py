"""Tests for the BM25 lexical index."""

from __future__ import annotations

from pattern_ladder.index.lexical import LexicalIndex


def test_finds_the_document_containing_the_jargon(lexical_index, problems):
    hits = lexical_index.search("monotonic stack", k=5)
    assert hits
    top = problems[hits[0][0]]
    assert "Monotonic Stack" in top.topics


def test_results_are_ordered_by_descending_score(lexical_index):
    hits = lexical_index.search("window", k=5)
    scores = [score for _doc, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_zero_scoring_documents_are_not_returned(lexical_index):
    """A BM25 score of zero means no query term occurs in the document.
    Returning it as a 'result' would pad the list with noise."""
    hits = lexical_index.search("parentheses", k=12)
    assert all(score > 0 for _doc, score in hits)


def test_k_larger_than_corpus_is_clamped(lexical_index, problems):
    """bm25s raises rather than clamping when k exceeds the corpus size."""
    hits = lexical_index.search("window", k=len(problems) * 10)
    assert len(hits) <= len(problems)


def test_stopword_only_query_returns_nothing(lexical_index):
    """Returning arbitrary documents here would be worse than returning none:
    the dense arm can still carry the query."""
    assert lexical_index.search("the and of", k=5) == []


def test_empty_query_returns_nothing(lexical_index):
    assert lexical_index.search("", k=5) == []


def test_non_positive_k_returns_nothing(lexical_index):
    assert lexical_index.search("window", k=0) == []


def test_stemming_matches_inflected_forms(lexical_index, problems):
    """'shrinking' must reach a document that says 'shrink'."""
    hits = lexical_index.search("shrinking windows", k=5)
    assert hits
    titles = {problems[doc].title for doc, _ in hits}
    assert any("Substring" in t or "Window" in t for t in titles)


def test_save_and_load_round_trip(lexical_index, tmp_path):
    lexical_index.save(tmp_path / "bm25")
    reloaded = LexicalIndex.load(tmp_path / "bm25")
    assert reloaded.search("monotonic stack", k=3) == lexical_index.search("monotonic stack", k=3)
