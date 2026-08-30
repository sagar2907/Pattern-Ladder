"""Cross-encoder reranking.

The retrievers score a query and a document independently and compare vectors.
A cross-encoder reads the pair together, so it can represent "this document
answers this query" rather than "these two texts are about similar things".
That distinction is what fixes the case where RRF surfaces a problem that
mentions the right words in the wrong role.

It costs a forward pass per candidate, which is why it runs over 50 documents
and not 2,830.
"""

from __future__ import annotations

from .. import config

_MODEL_CACHE: dict[str, object] = {}


def get_reranker(model_name: str = config.RERANKER_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import CrossEncoder

        _MODEL_CACHE[model_name] = CrossEncoder(model_name, device="cpu")
    return _MODEL_CACHE[model_name]


def rerank(
    query: str,
    documents: list[tuple[int, str]],
    *,
    model_name: str = config.RERANKER_MODEL,
    batch_size: int = 32,
) -> list[tuple[int, float]]:
    """Rescore (doc_id, text) pairs against the query.

    Returns (doc_id, score) sorted best first. Scores are the model's raw
    logits: unbounded, typically negative, and meaningful only *relative to
    each other within one query*. They are deliberately not squashed to [0,1] --
    a sigmoid would invite the UI to present them as confidence, which they
    are not.
    """
    if not documents:
        return []

    model = get_reranker(model_name)
    pairs = [(query, text) for _doc_id, text in documents]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    ranked = [
        (doc_id, float(score))
        for (doc_id, _text), score in zip(documents, scores, strict=True)
    ]
    # Tie-break on doc_id for determinism, as in fusion.
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked
