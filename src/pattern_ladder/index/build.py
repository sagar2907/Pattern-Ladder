"""Offline index construction.

Everything expensive happens here, once, and is cached to disk: corpus
normalisation, BM25 indexing, dense encoding, graph construction and community
detection. At query time the only model that runs is the cross-encoder, over 50
candidates. That split is what keeps a cold start near the cost of loading two
small models rather than re-deriving the whole pipeline.

Determinism, stated precisely, because the obvious version of the claim is
false. No clock is read, no unseeded randomness is used, and every collection
feeding a positional artefact is sorted. Within one environment, repeated runs
produce byte-identical corpus, BM25 index, embeddings and graph.

The *clustering* is reproducible only within a pinned dependency set. Louvain
is seeded, and given an identical graph and seed it still returned 472
communities under numpy 2.5.2 and 473 under numpy 2.4.6 -- same networkx, same
input, byte-identical embeddings. Its modularity comparisons come down to
floating-point ties often enough that a change in the numeric stack decides
some of them differently.

That is why .python-version exists. The lockfile carries numpy twice, keyed on
the Python version, so without a pinned interpreter `uv sync` resolves a
different numeric stack on 3.11 than on 3.12 and the family count moves. The
project is reproducible; it is reproducible *against a pinned environment*,
which is a weaker and more honest claim than the one this docstring used to
make.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .. import config
from ..data import Problem, build_corpus, load_corpus
from ..graph.build import add_knn_backfill, build_link_graph
from ..graph.build import describe as describe_graph
from ..graph.families import (
    Family,
    build_families,
    coherence,
    detect_families,
    link_related_families,
    tag_independence,
)
from ..graph.naming import describe_all
from .dense import DenseIndex
from .lexical import LexicalIndex


def build_all(
    paths: config.Paths | None = None,
    *,
    title_repeat: int = 3,
    knn_backfill: bool = True,
    describe_families: bool = False,
    progress=None,
) -> dict:
    """Build every artefact and return the manifest.

    Returns the manifest rather than writing-and-forgetting so that callers
    (tests, the sweep scripts) can assert on the measured numbers directly.
    """
    paths = paths or config.default_paths()
    paths.index.mkdir(parents=True, exist_ok=True)

    problems = build_corpus(paths, title_repeat=title_repeat)
    texts = [p.index_text for p in problems]

    lexical = LexicalIndex.build(texts)
    lexical.save(paths.bm25)

    dense = DenseIndex.build(texts)
    dense.save(paths.embeddings)

    # Read any previous families before overwriting them, so descriptions
    # already paid for survive a rebuild that was not about them.
    existing = None
    if paths.families.exists():
        try:
            existing = [Family(**row) for row in json.loads(
                paths.families.read_text(encoding="utf-8")
            )]
        except (OSError, ValueError, TypeError):
            existing = None

    families, link_stats, final_stats, described = derive_families(
        problems,
        dense.matrix,
        knn_backfill=knn_backfill,
        describe_families=describe_families,
        progress=progress,
        existing=existing,
    )

    paths.families.write_text(
        json.dumps([asdict(f) for f in families], ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = _manifest(
        problems, families, link_stats, final_stats, title_repeat, knn_backfill, described
    )
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def derive_families(
    problems: list[Problem],
    embeddings: np.ndarray,
    *,
    knn_backfill: bool = True,
    describe_families: bool = False,
    progress=None,
    existing=None,
    resolution: float = config.LOUVAIN_RESOLUTION,
    min_size: int = config.MIN_FAMILY_SIZE,
):
    """Graph construction, clustering and naming: everything after encoding.

    Split out from `build_all` so it can be exercised without downloading a
    corpus or loading a model. That separation is not cosmetic -- when this
    logic lived inline, `build_all` had no test at all, and a parameter that
    shadowed an imported function slipped through the suite untouched.

    Returns (families, link_only_stats, final_stats, described_count).
    """
    graph = build_link_graph(problems)
    link_stats = describe_graph(graph)
    if knn_backfill:
        graph = add_knn_backfill(graph, problems, embeddings)
    final_stats = describe_graph(graph)

    # Resolution and minimum size are parameters rather than constants read
    # from config, so this can be exercised on a corpus far smaller than the
    # real one -- at the production resolution a twelve-problem fixture
    # shatters into communities below the size threshold and yields nothing
    # to assert on.
    communities = detect_families(graph, resolution=resolution)
    families = build_families(communities, problems, min_size=min_size)
    # Cross-family links come from the curated graph, so this runs on the
    # finished families rather than during clustering.
    families = link_related_families(families, problems)

    # Descriptions are written once, here, and cached with the families. No
    # query ever waits on a model, and a build with no API key produces
    # identical artefacts minus this one optional field.
    if describe_families:
        descriptions = describe_all(families, problems, progress=progress)
        families = [
            replace(family, description=descriptions.get(family.family_id))
            for family in families
        ]
    elif existing is not None:
        families = _carry_forward_descriptions(families, existing)

    described = sum(1 for f in families if f.description)
    return families, link_stats, final_stats, described


def _carry_forward_descriptions(families, existing) -> list:
    """Reuse descriptions from a previous build where the family is unchanged.

    Descriptions cost one API call each, and a rebuild triggered for an
    unrelated reason -- adding a field, re-running after a config change --
    would otherwise discard all 137 of them silently. That happened once, and
    the only sign was a manifest counter dropping to zero.

    Matched on the exact member set rather than on family_id. Ids are
    positional over the sorted community list, so inserting or removing a
    single community renumbers everything after it; the members are what the
    description actually describes.
    """
    previous = {
        frozenset(family.members): family.description
        for family in existing
        if family.description
    }
    if not previous:
        return families
    return [
        family
        if family.description
        else replace(family, description=previous.get(frozenset(family.members)))
        for family in families
    ]


def _manifest(
    problems: list[Problem],
    families,
    link_stats,
    final_stats,
    title_repeat: int,
    knn_backfill: bool,
    described: int = 0,
) -> dict:
    covered = sum(f.size for f in families)
    return {
        "dataset": {
            "repo": config.DATASET_REPO,
            "file": config.DATASET_FILE,
            "revision": config.DATASET_REVISION,
        },
        "models": {
            "dense": config.DENSE_MODEL,
            "reranker": config.RERANKER_MODEL,
            "dense_dim": config.EMBEDDING_DIM,
        },
        "corpus": {
            "problems": len(problems),
            "title_repeat": title_repeat,
            "difficulty": _count(p.difficulty for p in problems),
        },
        "graph": {
            "knn_backfill": knn_backfill,
            "link_only": asdict(link_stats),
            "final": asdict(final_stats),
        },
        "families": {
            "count": len(families),
            "min_size": config.MIN_FAMILY_SIZE,
            "covered_problems": covered,
            "coverage_fraction": round(covered / len(problems), 4) if problems else 0.0,
            "largest": max((f.size for f in families), default=0),
            # Agreement with the tag taxonomy, over assigned problems only.
            # See families.tag_independence for why the whole-corpus version of
            # this number is misleading.
            "tag_independence": tag_independence(families, problems),
            "name_coherence": coherence(families, problems),
            "size_histogram": _size_bands(families),
            "described_by_model": described,
            "with_related_families": sum(1 for f in families if f.related),
            "duplicate_names": _duplicate_names(families),
        },
        "retrieval": {
            "candidates_per_retriever": config.CANDIDATES_PER_RETRIEVER,
            "fusion_pool": config.FUSION_POOL_SIZE,
            "rrf_k": config.RRF_K,
        },
    }


def _duplicate_names(families) -> int:
    """How many families share their name with another.

    The tag-derived name is not guaranteed unique -- four separate families
    came out as "Tree / Binary Tree" -- and a student shown a duplicated name
    cannot tell which pattern they are looking at. Tracked so the number cannot
    drift upward unnoticed.
    """
    counts: dict[str, int] = {}
    for family in families:
        counts[family.name] = counts.get(family.name, 0) + 1
    return sum(count for count in counts.values() if count > 1)


def _size_bands(families) -> dict[str, int]:
    """Family sizes in bands. A single mean would hide the failure mode that
    matters -- one enormous family alongside many tiny ones."""
    bands = {"5-9": 0, "10-19": 0, "20-39": 0, "40-79": 0, "80+": 0}
    for family in families:
        if family.size < 10:
            bands["5-9"] += 1
        elif family.size < 20:
            bands["10-19"] += 1
        elif family.size < 40:
            bands["20-39"] += 1
        elif family.size < 80:
            bands["40-79"] += 1
        else:
            bands["80+"] += 1
    return bands


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def load_all(paths: config.Paths | None = None):
    """Load cached artefacts. Raises if the index has not been built."""
    paths = paths or config.default_paths()
    missing = [
        str(p)
        for p in (paths.corpus, paths.embeddings, paths.bm25, paths.families)
        if not Path(p).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "index artefacts missing: "
            + ", ".join(missing)
            + " -- run `python -m scripts.build_index`"
        )

    problems = load_corpus(paths.corpus)
    lexical = LexicalIndex.load(paths.bm25)
    dense = DenseIndex.load(paths.embeddings)
    families_raw = json.loads(paths.families.read_text(encoding="utf-8"))

    _assert_artefacts_aligned(problems, lexical, dense)
    return problems, lexical, dense, families_raw


def _assert_artefacts_aligned(problems, lexical, dense) -> None:
    """Refuse to serve artefacts that do not describe the same corpus.

    The embedding matrix and the BM25 index are both *positional*: row 7 is
    problem 7. Nothing in the file formats records which corpus they were built
    from, so a half-finished rebuild, an interrupted `build_index`, or an
    embedding file left over from a different `title_repeat` produces four
    artefacts that load without complaint and disagree about what document 7
    is.

    That is the worst failure mode this system has. Every downstream number
    stays plausible -- results are returned, ladders are built, nothing raises
    -- and every one of them is wrong. When it does eventually fail it is as an
    IndexError from inside a matrix slice, thousands of lines from the cause.

    Checking three integers at load time converts that into an error that names
    the problem and the fix.
    """
    corpus_size = len(problems)
    embedding_rows = int(dense.matrix.shape[0])
    if embedding_rows != corpus_size:
        raise ValueError(
            f"embeddings describe {embedding_rows} problems but the corpus has "
            f"{corpus_size}; the cached index is stale. "
            "Run `python scripts/build_index.py` to rebuild."
        )

    indexed_docs = lexical.document_count
    if indexed_docs is not None and indexed_docs != corpus_size:
        raise ValueError(
            f"the BM25 index describes {indexed_docs} documents but the corpus "
            f"has {corpus_size}; the cached index is stale. "
            "Run `python scripts/build_index.py` to rebuild."
        )


def embeddings_only(paths: config.Paths | None = None) -> np.ndarray:
    paths = paths or config.default_paths()
    return np.load(paths.embeddings)
