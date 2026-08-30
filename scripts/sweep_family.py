"""Sweep graph parameters against ladder quality, not against graph shape.

The earlier sweep (scripts/sweep_graph.py) optimised coverage and family size,
because those were the only things measurable before an evaluation set existed.
They are proxies. What actually matters is whether the ladder a student is
shown comes from a family that describes their question, and that is what
eval/smoke_queries.json measures as family@1.

The trick that makes this affordable: retrieval does not depend on the graph.
BM25, the dense encoder and the cross-encoder produce the same ranked results
whatever the clustering is, so the twenty queries are run *once* and their
results cached. Each configuration then only rebuilds families and re-runs
family selection over those cached results, which is milliseconds rather than
the ~30 seconds a full re-evaluation would cost.

Run: python scripts/sweep_family.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder import config  # noqa: E402
from pattern_ladder.data import load_corpus  # noqa: E402
from pattern_ladder.engine import load_engine  # noqa: E402
from pattern_ladder.graph import build as graph_build  # noqa: E402
from pattern_ladder.graph.families import (  # noqa: E402
    build_families,
    coherence,
    detect_families,
    tag_independence,
)
from pattern_ladder.index.build import embeddings_only  # noqa: E402
from pattern_ladder.retrieval.search import SearchEngine  # noqa: E402
from pattern_ladder.understand.groq_client import understand  # noqa: E402

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "smoke_queries.json"
TOP_K = config.FAMILY_VOTE_DEPTH


def cache_retrieval(engine: SearchEngine) -> list[dict]:
    """Run every query once and keep the ranked slugs and parsed intent."""
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["queries"]
    cached = []
    for case in cases:
        intent = understand(case["query"], allow_network=False)
        response = engine.search(case["query"], intent, top_k=TOP_K)
        cached.append(
            {
                "case": case,
                "intent": intent,
                "slugs": [r.problem.slug for r in response.results],
                "query_vector": None,
            }
        )
    return cached


def score_config(problems, embeddings, cached, *, neighbours, min_sim, resolution, knn_weight):
    graph = graph_build.build_link_graph(problems)
    original = graph_build.KNN_WEIGHT
    graph_build.KNN_WEIGHT = knn_weight
    try:
        graph = graph_build.add_knn_backfill(
            graph, problems, embeddings, neighbours=neighbours, min_similarity=min_sim
        )
    finally:
        graph_build.KNN_WEIGHT = original

    communities = detect_families(graph, resolution=resolution)
    families = build_families(communities, problems)

    # A throwaway engine carrying the new families; the indexes are irrelevant
    # here because only family lookup and selection are exercised.
    engine = SearchEngine(problems, None, None, families)

    scored = hits = 0
    for row in cached:
        wanted = row["case"].get("expect_family_contains")
        if not wanted:
            continue
        # Same (problem, family) shape the engine votes over.
        ranked = [
            (engine._by_slug[slug], engine.family_for(slug))
            for slug in row["slugs"]
            if slug in engine._by_slug
        ]
        family = engine._select_family(ranked, row["intent"])
        if family is None:
            scored += 1
            continue
        haystack = f"{family.name} {' '.join(family.tags)}".lower()
        scored += 1
        hits += wanted.lower() in haystack

    sizes = [f.size for f in families]
    return {
        "neighbours": neighbours,
        "min_sim": min_sim,
        "resolution": resolution,
        "knn_weight": knn_weight,
        "families": len(families),
        "largest": max(sizes, default=0),
        "median_size": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        "coverage": round(sum(sizes) / len(problems), 4),
        "family_at_1": round(hits / scored, 4) if scored else None,
        "coherence": coherence(families, problems)["coherence"],
        "nmi_vs_tags": tag_independence(families, problems)["nmi"],
    }


def main() -> int:
    paths = config.default_paths()
    problems = load_corpus(paths.corpus)
    embeddings = embeddings_only(paths)

    engine = load_engine()
    cached = cache_retrieval(engine)
    del engine

    rows = []
    for neighbours, min_sim, resolution, knn_weight in itertools.product(
        [1, 2], [0.65, 0.75], [2.8, 3.5, 4.5, 6.0, 8.0], [0.45]
    ):
        rows.append(
            score_config(
                problems,
                embeddings,
                cached,
                neighbours=neighbours,
                min_sim=min_sim,
                resolution=resolution,
                knn_weight=knn_weight,
            )
        )

    rows.sort(key=lambda r: (-(r["family_at_1"] or 0), -r["coverage"]))
    Path("artifacts/sweep_family.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"nb={row['neighbours']} sim={row['min_sim']} res={row['resolution']:>4} "
            f"| fams={row['families']:3d} largest={row['largest']:3d} med={row['median_size']:3d} "
            f"cov={row['coverage']:.3f} family@1={row['family_at_1']} "
            f"coh={row['coherence']} nmi={row['nmi_vs_tags']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
