"""BM25 lexical retrieval over the problem corpus.

BM25 is the arm that catches exact jargon: a student who types "monotonic
stack" or "Dijkstra" wants documents containing those literal tokens, and a
dense encoder will happily return semantically adjacent problems that never
mention them. It is cheap, has no model to load, and fails in a way that is
easy to reason about.
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import Stemmer

# Tokenisation settings are persisted with the index. A query tokenised with
# different settings than the corpus silently retrieves nothing useful -- it
# does not error -- so these must travel together.
STOPWORDS = "en"
STEMMER_LANGUAGE = "english"

# BM25 term-frequency saturation and length normalisation. bm25s defaults to
# k1=1.5, b=0.75. Both are swept in scripts/sweep_retrieval.py against the
# smoke set; see config.BM25_K1 for what the sweep concluded.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class LexicalIndex:
    """Thin wrapper over bm25s that owns tokenisation consistency."""

    def __init__(self, retriever: bm25s.BM25, stemmer_language: str = STEMMER_LANGUAGE) -> None:
        self._retriever = retriever
        self._stemmer_language = stemmer_language
        self._stemmer = Stemmer.Stemmer(stemmer_language)

    @classmethod
    def build(
        cls, texts: list[str], *, k1: float = DEFAULT_K1, b: float = DEFAULT_B
    ) -> LexicalIndex:
        stemmer = Stemmer.Stemmer(STEMMER_LANGUAGE)
        tokens = bm25s.tokenize(
            texts, stopwords=STOPWORDS, stemmer=stemmer, show_progress=False
        )
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(tokens, show_progress=False)
        return cls(retriever)

    @property
    def document_count(self) -> int | None:
        """How many documents are indexed, or None if bm25s stops exposing it.

        Used to detect a cached index built from a different corpus. Returning
        None rather than raising keeps a future bm25s version from breaking
        loading over a consistency check.
        """
        try:
            return int(self._retriever.scores["num_docs"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return (corpus_index, score) pairs, best first.

        Queries are tokenised with `return_ids=False` so bm25s maps raw token
        strings through the *index's* vocabulary. Passing a Tokenized object
        instead makes the query carry its own vocabulary, and the two only
        happen to agree; decoupling them removes a whole class of silent
        mismatch.
        """
        if k <= 0:
            return []
        # bm25s raises if k exceeds the corpus size rather than clamping.
        k = min(k, self._retriever.scores["num_docs"])

        tokens = bm25s.tokenize(
            [query],
            stopwords=STOPWORDS,
            stemmer=self._stemmer,
            return_ids=False,
            show_progress=False,
        )
        if not tokens or not tokens[0]:
            # Query was entirely stopwords/punctuation. Returning [] lets the
            # dense arm carry the query rather than surfacing arbitrary docs.
            return []

        indices, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [
            (int(i), float(s))
            for i, s in zip(indices[0], scores[0], strict=True)
            if s > 0.0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory))
        (directory / "tokeniser.json").write_text(
            json.dumps({"stopwords": STOPWORDS, "stemmer": self._stemmer_language}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LexicalIndex:
        retriever = bm25s.BM25.load(str(directory), load_corpus=False)
        meta_path = directory / "tokeniser.json"
        language = STEMMER_LANGUAGE
        if meta_path.exists():
            language = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "stemmer", STEMMER_LANGUAGE
            )
        return cls(retriever, stemmer_language=language)
