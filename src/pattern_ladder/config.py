"""Central configuration: paths, model ids, and tuned constants.

Everything that another module might be tempted to hardcode lives here, so that
a change to (say) the fusion constant is a one-line edit with a comment
explaining the choice, rather than a magic number buried in a loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Locations ---------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Derived data. Never committed: `artifacts/` is gitignored and rebuilt by
# `python -m scripts.build_index`. Overridable so tests can point at a tmpdir.
ARTIFACTS_DIR = Path(os.environ.get("PATTERN_LADDER_ARTIFACTS", PROJECT_ROOT / "artifacts"))

RAW_DIR = ARTIFACTS_DIR / "raw"
INDEX_DIR = ARTIFACTS_DIR / "index"

CORPUS_PATH = INDEX_DIR / "corpus.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
BM25_DIR = INDEX_DIR / "bm25"
FAMILIES_PATH = INDEX_DIR / "families.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# --- Corpus source -----------------------------------------------------------

# The dataset's *default* parquet config is an instruction-tuning set with two
# columns (user_queries / expected_output) and none of the structured fields
# this project needs. The structured table only exists under raw_data/.
# Reaching for the default config here would silently give a corpus with no
# difficulty, no topics and no similar_questions.
DATASET_REPO = "Alishohadaee/leetcode-problems-dataset"
DATASET_FILE = "raw_data/leetcode_problems.json"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{DATASET_FILE}"

# Pinned so a corpus rebuild is reproducible. The dataset is a moving target
# only in principle -- it has not changed since 2025-05 -- but an unpinned
# revision means "reproducible build" is a claim we cannot actually make.
DATASET_REVISION = "main"

# --- Models ------------------------------------------------------------------

# 22.7M params, Apache-2.0. Chosen for CPU latency, not for leaderboard rank --
# and kept after a candidate that is measurably a *better encoder* made the
# system measurably worse.
#
# bge-small-en-v1.5 beats this model in isolation on the smoke set: dense-only
# hit@5 of 1.00 against 0.90. Swapped in, end-to-end hit@5 fell 0.95 -> 0.90,
# ladder accuracy 0.85 -> 0.75, and ladder-family accuracy 0.833 -> 0.583.
# Re-sweeping the graph parameters for it (scripts/sweep_family.py) recovered
# only to 0.75.
#
# The reason is that the embeddings are not only used for ranking: the graph
# backfill thresholds on raw cosine similarity, and bge-small's similarities
# sit higher, so the same 0.65 floor admitted 3,543 inferred edges instead of
# 1,320 and redrew the families. An encoder cannot be swapped in isolation
# here; whatever consumes its *distribution* has to be retuned with it.
# See scripts/compare_encoders.py.
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Canonical id. The commonly-cited "ms-marco-MiniLM-L-6-v2" (extra hyphen)
# still resolves via a Hub redirect, but naming the canonical repo avoids
# depending on that redirect continuing to exist.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

EMBEDDING_DIM = 384
# Documents per forward pass when building the index. This is the single
# number that decides whether the project fits its free hosting tier, and it
# costs nothing to get right.
#
# Peak resident memory during a full build, measured by sampling RSS every
# 20ms while encoding all 2,830 documents:
#
#     batch 64 -> 951 MB     batch 16 -> 687 MB
#     batch 32 -> 799 MB     batch  8 -> 686 MB
#
# Encoding took 110 seconds at every one of those settings, so the 264 MB
# saved between 64 and 16 is free. Against the ~1 GB ceiling of the target
# host, 64 left 54 MB of headroom before the web server's own footprint --
# which is to say none, and a first-run build that would very likely have
# been killed. Below 16 there is nothing further to win.
#
# The peak is not the embeddings themselves, which are 4.3 MB. It is the
# activations of one forward pass. Chunking the *input* was tried first, on
# the theory that the corpus list and its internal copies were the cost, and
# moved the peak by 1 MB.
ENCODE_BATCH = 16

MAX_SEQ_LENGTH = 256  # Problem statements are long; 256 tokens covers the
# discriminative part (statement + constraints) without paying for padding.

# --- Retrieval ---------------------------------------------------------------

# How many rungs of a family to show. A ladder is a plan for the next few
# sittings, not a reading list; a 60-problem family shown in full is the
# same overwhelming wall the flat list was.
LADDER_LENGTH = 7

# How many of a family's members stay eligible for the ladder after being
# ranked by relevance to the query. Wide enough that all three difficulty
# tiers are usually represented, narrow enough to drop the members that are
# in the family but not in the question.
LADDER_CANDIDATES = 20

# A family member stays on the ladder only if its similarity to the query is
# at least this fraction of the best member's. Swept in
# scripts/evaluate.py against ladder_hit over {0.0 ... 0.9}: removing the
# floor entirely scores 0.40, and 0.80 scores 0.85, with no change to hit@5 or
# family@1. Above 0.85 ladders start becoming too short to be a progression.
LADDER_RELEVANCE_RATIO = 0.8

# Floor on ladder length. Below this the relevance filter is relaxed and the
# most relevant members are taken regardless, because a one-rung ladder
# communicates less than a slightly loose one.
LADDER_MIN_RUNGS = 5

# How much a problem's approval (its like/dislike ratio) counts against its
# acceptance rate when ordering rungs inside one difficulty tier. Swept in
# scripts/evaluate.py against mean ladder approval, with ladder_hit and
# family@1 held as guardrails.
APPROVAL_WEIGHT = 0.5

CANDIDATES_PER_RETRIEVER = 100  # depth each arm contributes to fusion
FUSION_POOL_SIZE = 50  # survivors handed to the cross-encoder
DEFAULT_TOP_K = 10

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# and the de-facto default; it damps the influence of the very top ranks so a
# single confident-but-wrong retriever cannot dominate the fused list.
RRF_K = 60

# Characters of each candidate shown to the cross-encoder. A LeetCode
# statement puts the actual task first and the worked examples and
# constraints last, so the head of the text is the discriminative part and
# the tail is mostly boilerplate the model pays for and learns little from.
#
# Measured over 5 queries against full-text reranking: 600 characters is
# 2.2x faster (2.85s -> 1.32s per query), returns the identical top result
# every time, and keeps 86% of the top-10. Dropping to 300 reaches 0.55s but
# starts changing the top result, which is the one thing that must not move.
RERANK_TEXT_CHARS = 600

# Weight given to a family whose name or tags contain the parsed technique,
# on the same scale as the rank weights in SearchEngine._select_family (a
# top-ranked result contributes 1.0). Direct evidence that a family is about
# what the student asked for is worth about as much as one top hit.
TECHNIQUE_MATCH_BONUS = 1.0

# How many ranked results vote on which family the ladder comes from. Fixed
# rather than tied to the displayed result count, so changing how many rows are
# on screen cannot change which pattern the student is taught -- a discrepancy
# that showed up in the interface, where ten results chose the right family for
# the sliding-window query and five did not. Swept over {5, 8, 10, 15}: 8 is the
# shallowest depth that holds ladder accuracy at 0.85.
FAMILY_VOTE_DEPTH = 8

# Treat the cross-encoder as a third ranker to be fused, rather than as the
# final word. See SearchEngine.search for the measurements behind this.
RERANK_FUSION = True

# How much the reranker's opinion is worth relative to the retrieval
# ordering when the two are fused. Swept over {1.0, 1.5, 2.0, 3.0}: 1.0 and
# 1.5 both reach hit@5 = 0.95 with a perfect score on oblique phrasings,
# while 2.0 and above start reproducing the rerank-only regression
# (hit@5 0.90, oblique 0.83). Equal weight is also the least arbitrary
# choice among the two that tie.
RERANK_FUSION_WEIGHT = 1.0

# --- Graph -------------------------------------------------------------------

LOUVAIN_SEED = 42  # Louvain is stochastic; without a seed families would be
# renumbered on every rebuild and no result would be reproducible.

# Above 1.0, Louvain prefers smaller communities. At the default 1.0 the
# largest family came out at 260 problems -- a topic rather than a pattern,
# producing a ladder nobody can walk.
#
# This value was chosen twice. The first sweep optimised coverage and family
# size and settled on 2.8, because no measure of ladder *quality* existed
# yet. Once eval/smoke_queries.json could score whether the ladder came from
# a family that matched the question, 2.8 turned out to be badly
# under-split: at that setting a single 34-member community held three
# unrelated patterns (queue/stack design, heap/greedy, and
# anagram/sliding-window), so Minimum Window Substring lived in a family
# named 'Queue / Design'. Re-swept against family@1 in
# scripts/sweep_family.py, 8.0 lifts that score from 0.583 to 0.833 and
# improves name coherence, at no cost to coverage.
LOUVAIN_RESOLUTION = 8.0

# A family smaller than this cannot express a difficulty progression, so it is
# not shown as a ladder.
MIN_FAMILY_SIZE = 5

# How many neighbouring families to record per family, for the "what comes
# after this pattern" suggestion. Three is enough to offer a choice and few
# enough to fit beside a ladder without competing with it.
RELATED_FAMILIES = 3

# Under-connected problems are attached via dense-embedding neighbours. See
# graph/build.py for why this replaces the tag co-occurrence fallback.
#
# These four values are not defaults -- they are the sweep winner from
# scripts/sweep_graph.py over 108 configurations, chosen to maximise the
# share of the corpus sitting in a walkable family (5-80 members) subject to
# keeping at least 40 such families. Measured against the link-only graph
# they lift the share of the corpus sitting in a family from 49.2% to 86.3%
# and the family count from 60 to 137, while improving name coherence from
# 0.790 to 0.845 and ladder-family accuracy from 0.583 to 0.833.
#
# Agreement with the tag taxonomy does rise, from NMI 0.618 to 0.667. That
# is an honest cost of splitting finer: smaller communities line up with
# tags more often. At 0.667 roughly a third of the structure is still not
# explained by tags, so the families remain more than a tag rename -- but
# less emphatically than the coarser clustering, and that should not be
# overstated.
#
# The settings stay conservative: two neighbours at most, above a similarity
# floor, and only for nodes with almost no curated links. Bridging distinct
# families was the risk that forced these down from the 4 neighbours at
# 0.55 this started at; the high Louvain resolution now absorbs that risk,
# which is why two neighbours are affordable again.
KNN_NEIGHBOURS = 2
KNN_MIN_SIMILARITY = 0.65

# Only backfill nodes with at most this many curated edges. Backfilling
# well-connected nodes adds edges that bridge distinct families and merges
# them.
KNN_MAX_DEGREE = 1

# Require an inferred edge to clear BOTH the similarity floor and a shared
# topic tag. Similarity alone put unrelated problems at the head of ladders;
# see graph/build.add_knn_backfill for the failures this fixes.
KNN_REQUIRE_SHARED_TAG = True

# --- Query understanding -----------------------------------------------------

# Seconds between calls when describing families during an index build. The
# free tier allows 30 requests a minute; 2.2s keeps a comfortable margin, and a
# build step that trips the limit and silently leaves half the corpus
# undescribed is worse than one that takes a few minutes.
NAMING_THROTTLE_SECONDS = 2.2

# Cache model readings on disk, keyed by query, model and system prompt.
# Makes a repeated question give a repeated answer -- which greedy decoding
# alone does not guarantee -- and costs nothing to be wrong about, since a
# prompt change alters the key and orphans every stale entry.
PARSE_CACHE_ENABLED = True

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_TIMEOUT_SECONDS = 8.0
# Must cover reasoning tokens, not just the answer. This was 200, which
# failed *every* live call with a 400 json_validate_failed: the model spends
# a few hundred tokens reasoning before emitting the JSON and was being cut
# off mid-object. The bug was invisible for as long as the API went
# unexercised, because the failure degrades silently to the offline parser
# and the offline parser is good. With reasoning_effort="low" a completion
# runs ~50 tokens; 512 leaves room for the occasional long one.
GROQ_MAX_OUTPUT_TOKENS = 512


@dataclass(frozen=True)
class Paths:
    """Bundle of output paths, so tests can redirect all of them at once."""

    artifacts: Path
    raw: Path
    index: Path
    corpus: Path
    embeddings: Path
    bm25: Path
    families: Path
    manifest: Path

    @classmethod
    def under(cls, root: Path) -> Paths:
        index = root / "index"
        return cls(
            artifacts=root,
            raw=root / "raw",
            index=index,
            corpus=index / "corpus.json",
            embeddings=index / "embeddings.npy",
            bm25=index / "bm25",
            families=index / "families.json",
            manifest=index / "manifest.json",
        )


def default_paths() -> Paths:
    return Paths.under(ARTIFACTS_DIR)
