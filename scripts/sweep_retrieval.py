"""Sweep the retrieval-side parameters against the smoke set.

Covers the two settings that were previously taken on faith:

  title_repeat -- how many times a problem's title is repeated in its indexed
      text. BM25 has no notion of fields, so repetition is the only way to say
      a title match is worth more than a body match. It also changes the dense
      embedding, since both retrievers index the same string.

  k1 and b -- BM25 term-frequency saturation and length normalisation. bm25s
      defaults to 1.5 and 0.75. This corpus has unusually variable document
      lengths (a two-line statement next to a page of worked examples), which
      is exactly the situation where b matters.

Metric is recall of the expected problem in the fused candidate pool, plus
hit@5 without reranking. Both are measured without the cross-encoder, which
makes the sweep roughly 60x faster and is defensible here: these parameters
decide what enters the rerank pool, and the reranker cannot promote a document
that never made it in. The winner is then verified end-to-end with reranking.

Run: python scripts/sweep_retrieval.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder import config  # noqa: E402
from pattern_ladder.data import build_corpus  # noqa: E402
from pattern_ladder.index.dense import DenseIndex  # noqa: E402
from pattern_ladder.index.lexical import LexicalIndex  # noqa: E402
from pattern_ladder.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from pattern_ladder.understand.groq_client import understand  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "smoke_queries.json"


def main() -> int:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    intents = {c["query"]: understand(c["query"], allow_network=False) for c in cases}

    rows = []
    for title_repeat in (1, 2, 3, 5):
        # index_text depends on title_repeat, so both indexes are rebuilt.
        problems = build_corpus(title_repeat=title_repeat)
        texts = [p.index_text for p in problems]
        slug_to_row = {p.slug: i for i, p in enumerate(problems)}
        dense = DenseIndex.build(texts)

        for k1, b in itertools.product((0.9, 1.2, 1.5, 1.8), (0.3, 0.5, 0.75, 0.9)):
            lexical = LexicalIndex.build(texts, k1=k1, b=b)
            in_pool = top5 = 0
            for case in cases:
                query = case["query"]
                text = intents[query].to_search_text(query)
                fused = reciprocal_rank_fusion(
                    [
                        lexical.search(text, config.CANDIDATES_PER_RETRIEVER),
                        dense.search(text, config.CANDIDATES_PER_RETRIEVER),
                    ]
                )
                order = [doc for doc, _ in fused]
                target = slug_to_row.get(case["expect_slug"])
                in_pool += target in order[: config.FUSION_POOL_SIZE]
                top5 += target in order[:5]
            rows.append(
                {
                    "title_repeat": title_repeat,
                    "k1": k1,
                    "b": b,
                    "recall_at_pool": round(in_pool / len(cases), 3),
                    "hit_at_5_norerank": round(top5 / len(cases), 3),
                }
            )

    rows.sort(key=lambda r: (-r["recall_at_pool"], -r["hit_at_5_norerank"]))
    Path("artifacts/sweep_retrieval.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    for row in rows[:20]:
        print(
            f"title_repeat={row['title_repeat']} k1={row['k1']} b={row['b']} "
            f"| recall@pool={row['recall_at_pool']} hit@5={row['hit_at_5_norerank']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
