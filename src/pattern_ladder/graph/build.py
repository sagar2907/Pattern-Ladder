"""Construction of the problem-similarity graph.

Why this module is not just `G.add_edges_from(similar_questions)`
-----------------------------------------------------------------
The explicit similar-questions links are the best signal available: they are
curated, and the families they induce are genuinely not LeetCode tags. But
measured on the real corpus they are sparse -- 1,932 undirected edges over
2,830 problems, with 1,074 problems (38%) having no link at all, shattering the
graph into 1,248 components. Clustering that directly leaves roughly half the
corpus with no family, which means half of all searches return no ladder: the
product's entire output.

The obvious fallback -- connect problems that share two or more tags -- was
measured and rejected. It collapses the corpus into a handful of enormous
blobs whose members share exactly the tags they were joined on, i.e. it
reconstructs the tag taxonomy the project exists to go beyond.

What this module does instead is attach *under-connected* problems to their
nearest neighbours in dense-embedding space, at a lower edge weight than the
curated links. The weighting matters: Louvain optimises weighted modularity,
so a curated link counts for more evidence than an inferred one, and the
families stay anchored on the curated backbone rather than being redrawn by
embedding similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .. import config
from ..data import Problem

# A curated link is direct human evidence that two problems drill the same
# idea. An embedding neighbour is an inference. Louvain optimises *weighted*
# modularity, so this roughly 2:1 ratio is what keeps curated links deciding the
# family boundaries while the inferred ones only attach strays. Swept in
# scripts/sweep_graph.py rather than picked by feel.
LINK_WEIGHT = 1.0
KNN_WEIGHT = 0.45

# How many nearest neighbours to screen per node before applying the
# similarity floor and the shared-tag test. Wider than the number of edges
# actually kept, so a node is not left stranded when its closest couple of
# neighbours happen to fail the filters.
CANDIDATE_POOL_FACTOR = 8


@dataclass(frozen=True)
class GraphStats:
    """Measured properties of a built graph. Reported, never assumed."""

    nodes: int
    edges: int
    link_edges: int
    knn_edges: int
    isolated: int
    components: int
    largest_component: int
    mean_degree: float


def build_link_graph(problems: list[Problem]) -> nx.Graph:
    """Undirected graph over the curated similar-questions links."""
    graph = nx.Graph()
    # Add all nodes first: a problem with no links must still exist as an
    # isolated node, or the backfill below cannot find it.
    for problem in problems:
        graph.add_node(problem.slug)

    known = {p.slug for p in problems}
    for problem in problems:
        for target in problem.similar_slugs:
            if target in known and target != problem.slug:
                graph.add_edge(problem.slug, target, weight=LINK_WEIGHT, source="link")
    return graph


def add_knn_backfill(
    graph: nx.Graph,
    problems: list[Problem],
    embeddings: np.ndarray,
    *,
    neighbours: int = config.KNN_NEIGHBOURS,
    min_similarity: float = config.KNN_MIN_SIMILARITY,
    max_degree: int = config.KNN_MAX_DEGREE,
    require_shared_tag: bool = config.KNN_REQUIRE_SHARED_TAG,
) -> nx.Graph:
    """Attach under-connected problems via embedding neighbours.

    Only nodes with `degree <= max_degree` get new edges. Backfilling every
    node would add an order of magnitude more edges than the curated graph
    contains and the curated signal would be drowned out -- the families would
    become embedding clusters wearing a similar-questions costume.

    `require_shared_tag` makes the edge condition conjunctive: a candidate must
    be both textually near *and* share at least one topic tag. Similarity alone
    was measured to be too weak a signal on this corpus. Isolated problems are
    disproportionately short, recent, Easy ones whose text resembles anything
    else short and array-shaped, so pure-similarity backfill placed "Valid
    Anagram" in a shortest-path family and "Find Center of Star Graph" in a
    binary-tree family. Because those strays are Easy with high acceptance,
    they sorted to the *head* of the ladder, putting the least reliable members
    where a student looks first.

    Note this does not reduce the families to tags: the tag test only filters
    candidate edges that embedding similarity already proposed, and it is never
    a source of edges on its own. The tag-agreement figure reported by
    `families.tag_independence` is the guard on that claim.

    `embeddings` must be L2-normalised and row-aligned with `problems`.
    """
    if embeddings.shape[0] != len(problems):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows for {len(problems)} problems"
        )

    slugs = [p.slug for p in problems]
    needy = [i for i, slug in enumerate(slugs) if graph.degree(slug) <= max_degree]
    if not needy:
        return graph

    # One matrix product for all needy rows: (needy x dim) @ (dim x n).
    similarity = embeddings[needy] @ embeddings.T
    # Never allow a node to select itself as its own neighbour.
    for row, node_index in enumerate(needy):
        similarity[row, node_index] = -np.inf

    # Screen a pool wider than `neighbours`, then keep the first `neighbours`
    # survivors. Filtering a top-2 slice directly would mean a node whose two
    # nearest neighbours both fail the tag test gets no edge at all, even when
    # its third neighbour is a good match -- a silent loss of exactly the
    # under-connected problems this function exists to rescue.
    pool = min(max(neighbours * CANDIDATE_POOL_FACTOR, neighbours), similarity.shape[1] - 1)
    if pool <= 0:
        return graph

    topics = [frozenset(p.topics) for p in problems]
    top = np.argpartition(-similarity, pool - 1, axis=1)[:, :pool]

    for row, node_index in enumerate(needy):
        candidates = top[row]
        # Sort so that edge insertion order is deterministic.
        candidates = candidates[np.argsort(-similarity[row, candidates], kind="stable")]
        added = 0
        for candidate in candidates:
            if added >= neighbours:
                break
            score = float(similarity[row, candidate])
            if score < min_similarity:
                # Candidates are sorted by descending similarity, so once one
                # falls below the floor every later one does too.
                break
            target_index = int(candidate)
            if require_shared_tag and not (topics[node_index] & topics[target_index]):
                continue
            source, target = slugs[node_index], slugs[target_index]
            if graph.has_edge(source, target):
                continue
            graph.add_edge(source, target, weight=KNN_WEIGHT, source="knn")
            added += 1
    return graph


def describe(graph: nx.Graph) -> GraphStats:
    """Measure the graph. Every number here is computed, not estimated."""
    degrees = [d for _n, d in graph.degree()]
    components = list(nx.connected_components(graph))
    link_edges = sum(
        1 for _u, _v, d in graph.edges(data=True) if d.get("source") == "link"
    )
    knn_edges = graph.number_of_edges() - link_edges
    return GraphStats(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        link_edges=link_edges,
        knn_edges=knn_edges,
        isolated=sum(1 for d in degrees if d == 0),
        components=len(components),
        largest_component=max((len(c) for c in components), default=0),
        mean_degree=(sum(degrees) / len(degrees)) if degrees else 0.0,
    )
