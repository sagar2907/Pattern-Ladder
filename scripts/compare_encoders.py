"""Compare candidate dense encoders on this corpus and this evaluation set.

Published MTEB averages are the wrong basis for this decision. They are
averaged over dozens of tasks on prose that looks nothing like a LeetCode
statement, and the choice here is constrained by a CPU-only deployment with a
1GB memory ceiling. What matters is how a model ranks *these* documents for
*these* queries, and what it costs to run.

Measures, per model: time to encode the corpus, time to encode one query,
whether the expected problem reaches the candidate pool, and hit@5 before
reranking. Reranking is off so the comparison isolates the encoder.

Run: python scripts/compare_encoders.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder import config  # noqa: E402
from pattern_ladder.data import load_corpus  # noqa: E402
from pattern_ladder.index.dense import DenseIndex  # noqa: E402
from pattern_ladder.index.lexical import LexicalIndex  # noqa: E402
from pattern_ladder.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from pattern_ladder.understand.groq_client import understand  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "smoke_queries.json"

CANDIDATES = [
    # The incumbent: 22.7M params, Apache-2.0, the default recommendation for
    # CPU sentence embedding.
    "sentence-transformers/all-MiniLM-L6-v2",
    # 33M, MIT, consistently above MiniLM on MTEB retrieval specifically.
    "BAAI/bge-small-en-v1.5",
    # 47M, Apache-2.0, a recent purpose-built small retrieval model.
    "ibm-granite/granite-embedding-small-english-r2",
    # A static (non-transformer) model: no attention at inference, so it is
    # dramatically faster and lighter on CPU. Requires the optional `model2vec`
    # package; without it sentence-transformers does not fail cleanly, so it is
    # only attempted when the package is importable.
    *(
        ["minishlab/potion-retrieval-32M"]
        if importlib.util.find_spec("model2vec") is not None
        else []
    ),
]


def evaluate(model_name: str, problems, texts, lexical, cases, intents) -> dict:
    started = time.perf_counter()
    try:
        dense = DenseIndex.build(texts, model_name=model_name)
    except Exception as exc:  # noqa: BLE001 - a model that will not load is a result
        return {"model": model_name, "error": f"{type(exc).__name__}: {exc}"[:160]}
    corpus_seconds = time.perf_counter() - started

    slug_to_row = {p.slug: i for i, p in enumerate(problems)}

    query_times = []
    in_pool = top5 = dense_only_top5 = 0
    for case in cases:
        query = case["query"]
        text = intents[query].to_search_text(query)

        started = time.perf_counter()
        vector = dense.encode_query(text)
        query_times.append(time.perf_counter() - started)

        dense_hits = dense.search_vector(vector, config.CANDIDATES_PER_RETRIEVER)
        fused = reciprocal_rank_fusion(
            [lexical.search(text, config.CANDIDATES_PER_RETRIEVER), dense_hits]
        )
        order = [doc for doc, _ in fused]
        target = slug_to_row.get(case["expect_slug"])
        in_pool += target in order[: config.FUSION_POOL_SIZE]
        top5 += target in order[:5]
        dense_only_top5 += target in [doc for doc, _ in dense_hits[:5]]

    return {
        "model": model_name,
        "dim": int(dense.matrix.shape[1]),
        "corpus_encode_s": round(corpus_seconds, 1),
        "query_encode_ms": round(1000 * sum(query_times) / len(query_times), 2),
        "recall_at_pool": round(in_pool / len(cases), 3),
        "hit_at_5_hybrid": round(top5 / len(cases), 3),
        "hit_at_5_dense_only": round(dense_only_top5 / len(cases), 3),
    }


def main() -> int:
    problems = load_corpus(config.default_paths().corpus)
    texts = [p.index_text for p in problems]
    lexical = LexicalIndex.build(texts)
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    intents = {c["query"]: understand(c["query"], allow_network=False) for c in cases}

    out = Path("artifacts/compare_encoders.json")
    rows = []
    for model_name in CANDIDATES:
        rows.append(evaluate(model_name, problems, texts, lexical, cases, intents))
        # Persisted per candidate rather than at the end: one model that
        # hangs on load must not discard the results already measured.
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    header = (
        f"{'model':50s} {'dim':>4s} {'corpus_s':>9s} {'query_ms':>9s} "
        f"{'pool':>6s} {'hit@5':>6s} {'dense@5':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['model']:50s} FAILED: {row['error']}")
            continue
        print(
            f"{row['model']:50s} {row['dim']:>4d} {row['corpus_encode_s']:>9.1f} "
            f"{row['query_encode_ms']:>9.2f} {row['recall_at_pool']:>6.3f} "
            f"{row['hit_at_5_hybrid']:>6.3f} {row['hit_at_5_dense_only']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
