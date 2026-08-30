"""Dense retrieval over the problem corpus.

This is the arm that handles the situation the whole project exists for: a
student who cannot name the technique. "I keep failing problems where you
shrink a window from the left" shares almost no content words with the text of
a sliding-window problem, so BM25 scores it near zero. An embedding of the
sentence lands next to those problems anyway.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .. import config

# Loading a SentenceTransformer costs ~20s cold. Cache per-process so a
# Streamlit rerun does not pay it again; the model is stateless at inference.
_MODEL_CACHE: dict[str, object] = {}


def get_encoder(model_name: str = config.DENSE_MODEL):
    """Load (and memoise) the sentence encoder on CPU."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu")
        model.max_seq_length = config.MAX_SEQ_LENGTH
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def embedding_dimension(model_name: str = config.DENSE_MODEL) -> int:
    """The encoder's output width, needed to preallocate before encoding."""
    model = get_encoder(model_name)
    for attribute in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attribute, None)
        if getter is not None:
            return int(getter())
    return config.EMBEDDING_DIM


def encode(
    texts: list[str],
    *,
    batch_size: int = config.ENCODE_BATCH,
    model_name: str = config.DENSE_MODEL,
):
    """Embed texts to L2-normalised float32 vectors.

    Normalising at encode time means every later similarity is a plain dot
    product -- no per-query norm computation, and cosine similarity and inner
    product cannot drift apart in different call sites.

    The batch size is a memory decision and is the reason this deploys at all.
    See config.ENCODE_BATCH.
    """
    model = get_encoder(model_name)
    if not texts:
        return np.zeros((0, embedding_dimension(model_name)), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


class DenseIndex:
    """Exact brute-force search over a normalised embedding matrix.

    2,830 x 384 float32 is 4.3 MB and a full scan is a single BLAS call at
    ~1 ms. An ANN index (FAISS/HNSW) would add a dependency, a build step and
    an approximation error to save nothing measurable at this scale.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        model_name: str = config.DENSE_MODEL,
        encoder: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.model_name = model_name
        # Injectable so that callers holding a synthetic matrix -- the test
        # suite -- can supply a matching synthetic encoder. Without this the
        # index would always reach for the real model, which means every test
        # touching search would need a 90MB download and would compare 384-dim
        # query vectors against whatever dimension the fixture used.
        self._encoder = encoder

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a query into this index's own vector space."""
        if self._encoder is not None:
            vectors = np.asarray(self._encoder([text]), dtype=np.float32)
            return vectors[0]
        return encode([text], model_name=self.model_name)[0]

    @classmethod
    def build(cls, texts: list[str], *, model_name: str = config.DENSE_MODEL) -> DenseIndex:
        return cls(encode(texts, model_name=model_name), model_name=model_name)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0 or self.matrix.shape[0] == 0:
            return []
        return self.search_vector(self.encode_query(query), k)

    def search_vector(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.matrix @ vector.astype(np.float32)
        k = min(k, scores.shape[0])
        # argpartition is O(n) vs argsort's O(n log n); only the top-k need
        # ordering, and k is ~100 against n ~2830.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.matrix)

    @classmethod
    def load(cls, path: Path, model_name: str = config.DENSE_MODEL) -> DenseIndex:
        return cls(np.load(path), model_name=model_name)
