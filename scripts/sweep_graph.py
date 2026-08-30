"""Sweep the graph-construction parameters against a stated objective.

The parameters that control the graph (how many embedding neighbours to add,
how similar they must be, how much a curated link outweighs an inferred one,
and Louvain's resolution) trade two things off:

  * coverage  -- what fraction of the corpus ends up in a family at all, and
  * granularity -- whether those families are small enough to walk.

Neither alone is a good target. Maximising coverage merges everything into one
component and yields a single 2,000-problem "family", which is not a study
path. Maximising granularity yields hundreds of pairs, which is not a family.

The objective below therefore counts only problems that land in a family whose
size is inside a usable band, and reports NMI against the tag partition so the
"these are not LeetCode tags" claim stays falsifiable while tuning.

Run: python scripts/sweep_graph.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pattern_ladder import config  # noqa: E402
from pattern_ladder.data import load_corpus  # noqa: E402
from pattern_ladder.graph import build as graph_build  # noqa: E402
from pattern_ladder.graph.families import (  # noqa: E402
    build_families,
    coherence,
    detect_families,
    tag_independence,
)
from pattern_ladder.index.build import embeddings_only  # noqa: E402

# A family below MIN is too small to express a difficulty progression; above
# MAX it is a topic, not a pattern, and its ladder is too long to be a plan.
USABLE_MIN = config.MIN_FAMILY_SIZE
USABLE_MAX = 80


def evaluate(
    problems, embeddings, *, max_degree, neighbours, min_sim, knn_weight, resolution,
    require_shared_tag=True,
):
    graph = graph_build.build_link_graph(problems)
    original_weight = graph_build.KNN_WEIGHT
    graph_build.KNN_WEIGHT = knn_weight
    try:
        graph = graph_build.add_knn_backfill(
            graph,
            problems,
            embeddings,
            neighbours=neighbours,
            min_similarity=min_sim,
            max_degree=max_degree,
            require_shared_tag=require_shared_tag,
        )
    finally:
        graph_build.KNN_WEIGHT = original_weight

    communities = detect_families(graph, resolution=resolution)
    families = build_families(communities, problems, min_size=USABLE_MIN)

    usable = [f for f in families if USABLE_MIN <= f.size <= USABLE_MAX]
    usable_covered = sum(f.size for f in usable)

    return {
        "max_degree": max_degree,
        "neighbours": neighbours,
        "min_sim": min_sim,
        "knn_weight": knn_weight,
        "resolution": resolution,
        "edges": graph.number_of_edges(),
        "knn_edges": graph.number_of_edges() - graph_build.describe(graph).link_edges,
        "families": len(families),
        "usable_families": len(usable),
        "largest": max((f.size for f in families), default=0),
        "median_size": sorted(f.size for f in families)[len(families) // 2] if families else 0,
        "usable_coverage": round(usable_covered / len(problems), 4),
        "all_coverage": round(sum(f.size for f in families) / len(problems), 4),
        "nmi_vs_tags": tag_independence(families, problems)["nmi"],
        "coherence": coherence(families, problems)["coherence"],
        "modularity": round(nx.community.modularity(graph, communities, weight="weight"), 4),
    }


def main() -> int:
    paths = config.default_paths()
    problems = load_corpus(paths.corpus)
    embeddings = embeddings_only(paths)

    grid = itertools.product(
        [0, 1],                    # max_degree: isolated only, or degree<=1 too
        [1, 2, 3],                 # neighbours kept per node
        [0.65, 0.75, 0.82],        # min_similarity floor
        [0.25, 0.45],              # knn_weight relative to a curated link
        [1.6, 2.0, 2.4, 2.8],      # louvain resolution
    )

    rows = []
    for max_degree, neighbours, min_sim, knn_weight, resolution in grid:
        rows.append(
            evaluate(
                problems,
                embeddings,
                max_degree=max_degree,
                neighbours=neighbours,
                min_sim=min_sim,
                knn_weight=knn_weight,
                resolution=resolution,
            )
        )

    # Baseline: no backfill at all, which is what the brief proposed.
    baseline_graph = graph_build.build_link_graph(problems)
    baseline_comms = detect_families(baseline_graph)
    baseline_fams = build_families(baseline_comms, problems, min_size=USABLE_MIN)
    baseline_usable = [f for f in baseline_fams if USABLE_MIN <= f.size <= USABLE_MAX]
    baseline = {
        "config": "link-only (no backfill)",
        "coherence": coherence(baseline_fams, problems)["coherence"],
        "families": len(baseline_fams),
        "usable_families": len(baseline_usable),
        "largest": max((f.size for f in baseline_fams), default=0),
        "usable_coverage": round(sum(f.size for f in baseline_usable) / len(problems), 4),
        "nmi_vs_tags": tag_independence(baseline_fams, problems)["nmi"],
    }

    rows.sort(key=lambda r: -r["usable_coverage"])
    print(json.dumps({"baseline": baseline, "top": rows[:15]}, indent=2))
    Path("artifacts/sweep_graph.json").write_text(
        json.dumps({"baseline": baseline, "rows": rows}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
