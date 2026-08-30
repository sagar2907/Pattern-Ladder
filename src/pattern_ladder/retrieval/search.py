"""The search engine: one entry point that runs the whole pipeline.

    understand -> retrieve (BM25 + dense) -> fuse (RRF) -> rerank -> expand

The ordering is load-bearing and not arbitrary:

* Both retrievers run over the *whole* corpus, because each is there to cover
  the other's blind spot. Running one over the other's output would inherit the
  first one's misses.
* Fusion happens before reranking so the cross-encoder sees candidates neither
  arm alone would have ranked highly.
* The cross-encoder runs on 50 documents, not 2,830. It is ~50x slower per
  document than a dot product, so scoring the full corpus would take it from a
  sub-second step to a minute.
* Family expansion happens last, on survivors only, because a ladder is only
  worth building for a problem that is actually being recommended.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..data import Problem
from ..explain import explain, match_kind
from ..graph.families import Family, choose_start, order_ladder
from ..index.dense import DenseIndex
from ..index.lexical import LexicalIndex
from ..understand.schema import Intent
from .fusion import reciprocal_rank_fusion
from .rerank import rerank


@dataclass
class SearchResult:
    problem: Problem
    rerank_score: float
    fusion_rank: int
    match: str
    family: Family | None
    reason: str


@dataclass
class Ladder:
    """A difficulty-ordered slice of one family, centred on the student."""

    family: Family
    rungs: list[Problem]
    start_here: str | None
    truncated_from: int
    # Families this one leads on to, strongest link first. Answers the
    # question that follows a finished ladder: which pattern comes next.
    related_families: list[Family] = field(default_factory=list)
    # slug -> title of an earlier rung that LeetCode lists it as similar to.
    # This is what lets the ladder answer "why does this one follow that one"
    # with a curated fact rather than an inference. Kept as a side map rather
    # than folded into the rungs so that callers which only want the ordered
    # problems are unaffected.
    follows_from: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResponse:
    query: str
    intent: Intent
    results: list[SearchResult] = field(default_factory=list)
    ladder: Ladder | None = None
    # Populated only when a stage was skipped, so the UI can say so rather than
    # implying a full pipeline ran.
    notes: list[str] = field(default_factory=list)


class SearchEngine:
    """Holds the loaded indexes. Construct once; query many times."""

    def __init__(
        self,
        problems: list[Problem],
        lexical: LexicalIndex,
        dense: DenseIndex,
        families: list[Family],
    ) -> None:
        self.problems = problems
        self.lexical = lexical
        self.dense = dense
        self.families = families
        self._by_slug = {p.slug: p for p in problems}
        # slug -> family, so expansion is a dict lookup rather than a scan over
        # 86 families for each of 10 results.
        self._family_of: dict[str, Family] = {
            slug: family for family in families for slug in family.members
        }
        self._family_by_id: dict[int, Family] = {f.family_id: f for f in families}
        # slug -> row in the embedding matrix, so ladder relevance is a slice
        # rather than a re-encode.
        self._row_of: dict[str, int] = {p.slug: i for i, p in enumerate(problems)}

    def family_for(self, slug: str) -> Family | None:
        return self._family_of.get(slug)

    def search(
        self,
        query: str,
        intent: Intent,
        *,
        top_k: int = config.DEFAULT_TOP_K,
        pool_size: int = config.FUSION_POOL_SIZE,
        use_reranker: bool = True,
    ) -> SearchResponse:
        response = SearchResponse(query=query, intent=intent)

        # A query with no letters or digits carries no information, but the
        # dense encoder will still embed it somewhere and return that point's
        # nearest neighbours -- so an empty string used to produce five
        # confidently-ranked problems and a full ladder. The interface happens
        # to guard against this, but the engine is the public surface and must
        # not depend on its callers being careful.
        #
        # Deliberately narrow: only input with no alphanumeric character at all
        # is rejected. A bare number could be a problem id, and a single letter
        # could be a real if hopeless query; neither is nonsense in the way
        # that "" and "!!!" are.
        if not any(character.isalnum() for character in query):
            response.notes.append("the query contains nothing to search for")
            return response

        # The retrievers see the *expanded* query. Query understanding earns
        # its place here: a technique name recovered from loose phrasing is
        # exactly the jargon BM25 needs and the raw sentence never contains.
        search_text = intent.to_search_text(query)

        # Encoded once and reused: the dense arm needs it, and so does ladder
        # relevance below. Calling dense.search() with a string would re-encode.
        query_vector = self.dense.encode_query(search_text)

        lexical_hits = self.lexical.search(search_text, config.CANDIDATES_PER_RETRIEVER)
        dense_hits = self.dense.search_vector(query_vector, config.CANDIDATES_PER_RETRIEVER)

        if not lexical_hits and not dense_hits:
            response.notes.append("no candidates matched the query")
            return response
        if not lexical_hits:
            response.notes.append("lexical retrieval returned nothing; dense only")

        fused = reciprocal_rank_fusion([lexical_hits, dense_hits])

        lexical_ids = {doc_id for doc_id, _ in lexical_hits}
        dense_ids = {doc_id for doc_id, _ in dense_hits}

        # Difficulty filtering happens on the fused pool rather than after
        # reranking: filtering afterwards would routinely leave fewer than
        # top_k results, because the reranker's best candidates cluster in
        # whichever band the query resembles.
        candidates = [doc_id for doc_id, _score in fused]
        if intent.difficulty:
            wanted = intent.difficulty.lower()
            filtered = [
                doc_id
                for doc_id in candidates
                if self.problems[doc_id].difficulty.lower() == wanted
            ]
            # If the band is empty, ignore it rather than returning nothing --
            # an over-specific parse should not silently erase all results.
            if filtered:
                candidates = filtered
            else:
                response.notes.append(
                    f"no {intent.difficulty} problems among candidates; difficulty ignored"
                )

        pool = candidates[:pool_size]
        fusion_rank = {doc_id: rank for rank, doc_id in enumerate(pool)}

        if use_reranker and pool:
            reranked = rerank(
                search_text,
                [
                    (doc_id, self.problems[doc_id].index_text[: config.RERANK_TEXT_CHARS])
                    for doc_id in pool
                ],
            )
            if config.RERANK_FUSION:
                # Fuse the reranker's ordering with the retrieval ordering
                # rather than letting it overwrite it. Measured on the smoke
                # set, letting the cross-encoder decide outright *lowered*
                # hit@5 from 0.90 to 0.85, and on obliquely-phrased queries --
                # the ones this project exists to serve -- from 1.00 to 0.83.
                # It also sharpened precision, pulling the mean rank of found
                # problems from 1.56 to 1.06. Those are both real, which makes
                # this a trade rather than an upgrade, and RRF is how the rest
                # of this pipeline already resolves exactly that kind of
                # disagreement between two rankers with incomparable scores.
                scored_pairs = reciprocal_rank_fusion(
                    [[(doc_id, 0.0) for doc_id in pool], reranked],
                    weights=[1.0, config.RERANK_FUSION_WEIGHT],
                )
                rerank_scores = dict(reranked)
                scored = [(doc_id, rerank_scores.get(doc_id, 0.0)) for doc_id, _ in scored_pairs]
            else:
                scored = reranked
        else:
            if pool:
                response.notes.append("reranker disabled; fusion order shown")
            # Preserve fusion order with a descending pseudo-score so callers
            # can treat the two paths identically.
            scored = [(doc_id, -float(rank)) for rank, doc_id in enumerate(pool)]

        anchor = self.problems[scored[0][0]] if scored else None
        anchor_family = self.family_for(anchor.slug) if anchor else None

        # Family voting reads a fixed depth, independent of how many results
        # the caller chose to display. Without this the ladder changes when a
        # user moves the results slider, which is indefensible: the number of
        # rows on screen is a presentation choice and must not decide which
        # pattern the student is taught. It also showed up as a real
        # discrepancy -- at ten results the sliding-window query picked the
        # correct family, at five it did not.
        voting = [
            (self.problems[doc_id], self.family_for(self.problems[doc_id].slug))
            for doc_id, _score in scored[: config.FAMILY_VOTE_DEPTH]
        ]

        for doc_id, score in scored[:top_k]:
            problem = self.problems[doc_id]
            family = self.family_for(problem.slug)
            kind = match_kind(doc_id in lexical_ids, doc_id in dense_ids)
            response.results.append(
                SearchResult(
                    problem=problem,
                    rerank_score=score,
                    fusion_rank=fusion_rank.get(doc_id, -1),
                    match=kind,
                    family=family,
                    reason=explain(
                        problem,
                        kind=kind,
                        family=family,
                        anchor=anchor,
                        shared_family_with_anchor=(
                            family is not None
                            and anchor_family is not None
                            and family.family_id == anchor_family.family_id
                        ),
                        technique=intent.technique,
                    ),
                )
            )

        response.ladder = self._build_ladder(voting, intent, query_vector)
        if response.ladder is None and response.results:
            response.notes.append(
                "top results are not in a discovered family; no ladder available"
            )
        return response

    def _select_family(self, ranked: list[tuple[Problem, Family | None]], intent: Intent):
        """Choose which family the ladder is built from.

        Anchoring on the top-ranked result alone was wrong, and wrong in a way
        that showed up on the project's own example query. "Shrink a window
        from the left" ranks *Sliding Window Maximum* first, but that problem's
        discovered family is "Queue / Design" -- it is a monotonic-deque problem
        as much as a window one. The ladder that followed was Queue problems,
        including Valid Anagram and Group Anagrams. Every individual step had
        behaved correctly; the composition was still nonsense, because one
        result's family is a single noisy sample of what the query is about.

        Two signals are combined instead. Consensus across the ranked results,
        weighted by rank, so a family that several good hits agree on outranks
        one that a single hit happens to sit in. And a direct match between the
        parsed technique and the family's own name or tags, which is the
        strongest available evidence that a family is about the thing asked
        for -- and which is only available because the query was parsed into a
        technique in the first place.
        """
        if not ranked:
            return None

        by_id: dict[int, Family] = {}
        scores: dict[int, float] = {}
        for rank, (_problem, family) in enumerate(ranked):
            if family is None:
                continue
            by_id[family.family_id] = family
            scores[family.family_id] = scores.get(family.family_id, 0.0) + 1.0 / (rank + 1)

        if not scores:
            return None

        if intent.technique:
            technique = intent.technique.lower().strip()
            for family_id, family in by_id.items():
                haystack = f"{family.name} {' '.join(family.tags)}".lower()
                if technique and technique in haystack:
                    scores[family_id] += config.TECHNIQUE_MATCH_BONUS

        # A third signal was tried here and removed: scoring each family by
        # the mean similarity of its best members to the query. It sounds
        # like it should help and it did not -- family@1 fell from 0.583 to
        # 0.500 across every weight from 1 to 8. Families are broad enough
        # that their best few members are similar to almost any query in the
        # neighbourhood, so the term added variance without discrimination.
        #
        # Tie-break on family_id so the choice is reproducible.
        best_id = max(sorted(scores), key=lambda fid: scores[fid])
        return by_id[best_id]

    def _build_ladder(
        self, ranked: list[tuple[Problem, Family | None]], intent: Intent, query_vector
    ) -> Ladder | None:
        """Build a ladder: the family's most relevant problems, in study order.

        Selection and ordering are deliberately two separate steps, applied in
        that order.

        Ordering a whole family by difficulty alone was not enough. Families
        run to 60 members and are not perfectly pure, so the highest-acceptance
        problems in a tier are frequently the ones least related to the query.
        On "shrink a window from the left" the sliding-window family's first
        two Medium rungs came out as *Maximum Binary Tree* and *Maximum
        Difference Between Node and Ancestor*: genuinely in the family, and
        genuinely useless as the next step.

        So relevance to the query selects *which* members are eligible, and
        difficulty then orders the survivors. Doing it the other way round --
        sorting by relevance overall -- would produce a good list that is not a
        ladder, because a ladder's whole value is that it ascends.
        """
        family = self._select_family(ranked, intent)
        if family is None:
            return None

        members = [(self._by_slug[s], s) for s in family.members if s in self._by_slug]
        if not members:
            return None

        # Relevance of every family member to this query, by dot product
        # against the embedding matrix rows we already hold.
        rows = [self._row_of[slug] for _p, slug in members]
        similarity = self.dense.matrix[rows] @ query_vector

        # Named distinctly from the `ranked` parameter, which is the ranked
        # *results*; this is the family's members ranked by relevance. Reusing
        # the name worked only because _select_family had already been called.
        by_relevance = sorted(
            zip(members, similarity, strict=True),
            key=lambda item: (-float(item[1]), item[0][1]),
        )

        # Keep only members close to the best one. A fixed top-N was not
        # enough: a 27-member family whose 20 most-similar members still
        # include a dozen unrelated Easy problems hands those to the difficulty
        # sort, which promotes them to the first rungs. On "detect whether a
        # linked list has a cycle" that produced a ladder made entirely of
        # digit-arithmetic problems while the three correct linked-list
        # problems sat in the same family, unused.
        #
        # A relative floor rather than an absolute one, because similarity
        # scales differ per query: an absolute cut would empty the ladder for
        # broadly-phrased queries and never bite on narrow ones.
        # The guard on `best_score > 0` is load-bearing. Taking a fraction of a
        # non-positive score inverts the comparison: 0.8 * -0.05 is -0.04,
        # which is *greater* than -0.05, so the best member fails its own floor
        # and the filter starts keeping the least similar members instead of
        # the most. This is reachable on the real corpus -- one family in 1,507
        # query-family pairs probed had a negative best similarity. When
        # nothing in the family is positively similar there is no signal to
        # threshold on, so the top-N by relevance stands unfiltered.
        best_score = float(by_relevance[0][1]) if by_relevance else 0.0
        floor = (
            best_score * config.LADDER_RELEVANCE_RATIO
            if best_score > 0.0
            else float("-inf")
        )
        eligible = [
            problem
            for (problem, _slug), score in by_relevance[: config.LADDER_CANDIDATES]
            if float(score) >= floor
        ]
        # A ladder of one rung is not a ladder. When the floor is that
        # aggressive the family simply does not hold many problems about this
        # query, and the honest response is the most relevant few rather than
        # nothing -- the alternative is showing a student a single Hard problem
        # and calling it a progression.
        if len(eligible) < config.LADDER_MIN_RUNGS:
            eligible = [
                problem for (problem, _slug), _score in by_relevance[: config.LADDER_MIN_RUNGS]
            ]

        if intent.difficulty and intent.mode == "single":
            wanted = intent.difficulty.lower()
            narrowed = [p for p in eligible if p.difficulty.lower() == wanted]
            # A single-difficulty ladder is not a ramp, so only narrow when the
            # student asked for one problem rather than a progression.
            if narrowed:
                eligible = narrowed

        rungs = order_ladder(eligible)[: config.LADDER_LENGTH]
        return Ladder(
            family=family,
            rungs=rungs,
            start_here=choose_start(rungs),
            truncated_from=len(family.members),
            follows_from=_curated_links_within(rungs),
            related_families=[
                self._family_by_id[fid]
                for fid in family.related
                if fid in self._family_by_id
            ],
        )


def _curated_links_within(rungs: list[Problem]) -> dict[str, str]:
    """Map each rung to an earlier rung it is curated as similar to.

    This is the ladder's answer to "why does this one follow that one". The
    honest answer is only available when a person already recorded the
    relationship, so a rung with no such link gets no claim rather than an
    invented one -- which is why the map is sparse and callers must handle a
    missing entry.

    Only *earlier* rungs are considered, because the claim being made is about
    what prepares you for this problem, and a link to something further up the
    ladder does not.
    """
    links: dict[str, str] = {}
    for position, rung in enumerate(rungs):
        for earlier in rungs[:position]:
            if earlier.slug in rung.similar_slugs or rung.slug in earlier.similar_slugs:
                links[rung.slug] = earlier.title
                break
    return links
