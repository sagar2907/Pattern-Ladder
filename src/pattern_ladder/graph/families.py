"""Community detection, family naming, and ladder construction.

A "family" is a Louvain community over the similarity graph. A "ladder" is that
family ordered so a student can walk it: easiest first, and within one
difficulty tier by `rung_quality`, which blends how approachable a problem is
with how well regarded it is. Those two are close to independent -- the corpus
correlation is +0.14 -- so ordering on approachability alone is blind to
whether a problem was worth solving at all.

This module also records which families lead on to which, from curated links
that cross a family boundary. A ladder answers "what next within this
pattern"; `link_related_families` answers the question after it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace

import networkx as nx

from .. import config
from ..data import Problem

# Words that appear in so many LeetCode titles that they cannot distinguish one
# family from another. Naming a family "Number" tells a student nothing.
_GENERIC_TITLE_TOKENS = frozenset(
    {
        "the", "of", "in", "a", "an", "and", "or", "to", "with", "for", "from",
        "all", "two", "i", "ii", "iii", "iv", "number", "count", "find",
        "maximum", "minimum", "given", "make", "is", "by", "on", "at", "into",
        "that", "you", "your", "it", "its", "be", "can", "are", "as", "if",
    }
)

# Separator between the parts of a generated family name. A middle dot reads as
# "and also" without implying the parts are ordered or hierarchical.
NAME_SEPARATOR = " / "


@dataclass(frozen=True)
class Family:
    """A discovered problem family and its ladder.

    `members` is the community in ladder order: easiest first, and within a
    difficulty tier the most-accepted problem first.
    """

    family_id: int
    name: str
    tags: list[str]
    members: list[str]  # slugs, in ladder order
    size: int
    start_here: str | None = None
    difficulty_spread: dict[str, int] = field(default_factory=dict)
    # An optional model-written phrase describing the shared technique. Purely
    # additive: `name` above is deterministic and always present, so nothing
    # depends on this existing, and every metric that scores against family
    # names scores against the same strings whether or not it is set.
    description: str | None = None
    # family_ids of the families this one is most linked to, strongest first.
    # See `link_related_families` for what "linked" means and why only curated
    # edges count.
    related: list[int] = field(default_factory=list)

    @property
    def headline(self) -> str:
        """What to show a student: the description if there is one, else the name."""
        return self.description or self.name

    def headline_for(self, technique: str | None) -> str:
        """The headline to show for a query about `technique`.

        A community can hold more than one technique. One is tagged both
        Sliding Window and Prefix Sum, and its description -- a single line
        written once per family, with no knowledge of any query -- says "use
        prefix sums to compute subarray sums quickly". Leading with that for a
        student who asked about shrinking a window names the wrong pattern,
        which is the one thing this system exists to get right, and the
        deterministic name that does carry Sliding Window was demoted beneath
        it.

        So when the family covers the technique asked about and the description
        instead names a *different tag of this same family*, the name leads and
        the description follows. The description is not discarded -- the
        interface still shows it -- because it is usually the more informative
        of the two.

        The sibling-tag condition is what makes this safe, and merely asking
        whether the description omits the technique does not work. A family
        named "Linked / Linked List / Recursion" is described as "reordering
        nodes by pointer manipulation", which never says "linked list" and is
        plainly the better line of the two; demoting it would make the display
        worse on exactly the queries it already handled well. Naming a sibling
        tag is the signal that the description picked one technique out of
        several rather than paraphrasing the only one.
        """
        if not technique or not self.description:
            return self.headline
        wanted = technique.lower().strip()
        described = self.description.lower()
        if not wanted or wanted in described:
            return self.headline
        covered = f"{self.name} {' '.join(self.tags)}".lower()
        if wanted not in covered:
            return self.headline
        names_a_sibling = any(
            tag.lower() in described for tag in self.tags if tag.lower() != wanted
        )
        return self.name if names_a_sibling else self.headline


def detect_families(
    graph: nx.Graph,
    *,
    seed: int = config.LOUVAIN_SEED,
    resolution: float = config.LOUVAIN_RESOLUTION,
) -> list[set[str]]:
    """Louvain communities, ordered largest-first and deterministically.

    Louvain is stochastic: it visits nodes in a randomised order and returns a
    different partition per run. Seeding fixes the partition, but the order of
    the returned list is still arbitrary, so it is sorted here. Family ids are
    positional, so an unsorted list would renumber every family whenever the
    corpus changed by a single problem.
    """
    communities = nx.community.louvain_communities(
        graph, weight="weight", seed=seed, resolution=resolution
    )
    return sorted(communities, key=lambda c: (-len(c), min(c)))


def _distinctive_tags(
    members: list[Problem], corpus_tag_freq: Counter, corpus_size: int, limit: int = 3
) -> list[str]:
    """Tags over-represented in this family relative to the corpus.

    Ranking by raw count would name almost every family "Array". Ranking by
    lift alone would pick a tag that a single member happens to carry. The
    product of lift and in-family count balances distinctiveness against
    coverage.
    """
    if not members:
        return []
    family_freq = Counter(tag for m in members for tag in m.topics)
    scored: list[tuple[float, str]] = []
    for tag, count in family_freq.items():
        share_in_family = count / len(members)
        share_in_corpus = corpus_tag_freq.get(tag, 0) / max(corpus_size, 1)
        if share_in_corpus <= 0:
            continue
        lift = share_in_family / share_in_corpus
        scored.append((lift * count, tag))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [tag for _score, tag in scored[:limit]]


def _distinctive_title_token(
    members: list[Problem], corpus_token_freq: Counter, corpus_size: int
) -> str | None:
    """The word that most distinguishes these titles from the corpus at large.

    Supplies the concrete noun a tag list misses -- Palindrome, Subarray,
    Stock -- which is what makes a family name readable rather than taxonomic.
    """
    if len(members) < 3:
        return None

    family_tokens: Counter = Counter()
    for member in members:
        for token in _title_tokens(member.title):
            family_tokens[token] += 1

    best: tuple[float, str] | None = None
    for token, count in family_tokens.items():
        # Require the token in at least a quarter of members, else it is noise
        # from one or two titles that happen to share an unusual word.
        if count < max(3, len(members) * 0.25):
            continue
        share_in_family = count / len(members)
        share_in_corpus = corpus_token_freq.get(token, 0) / max(corpus_size, 1)
        if share_in_corpus <= 0:
            continue
        lift = share_in_family / share_in_corpus
        if best is None or lift > best[0]:
            best = (lift, token)
    return best[1].capitalize() if best else None


def _title_tokens(title: str) -> set[str]:
    """Content words of a title, deduplicated within the title.

    Deduplicating per title stops a repeated word inflating the family count
    above the membership threshold on the strength of one problem.
    """
    tokens = set()
    for raw in title.split():
        token = raw.lower().strip(".,()[]:;'\"")
        if token and len(token) > 2 and token not in _GENERIC_TITLE_TOKENS:
            tokens.add(token)
    return tokens


def rung_quality(problem: Problem, *, approval_weight: float = config.APPROVAL_WEIGHT) -> float:
    """How good a next rung this problem is, within its difficulty tier.

    Blends two things that sound alike and are not. Acceptance rate is
    approachability: how often people who try it succeed. Approval is whether
    the people who solved it thought it was worth solving. Across the corpus
    they correlate at only +0.14, so ordering on acceptance alone is blind to
    an entire dimension -- and blind in a way that actively misleads, because a
    badly-regarded problem is often badly regarded *for being easy to guess*,
    which inflates its acceptance and floats it to the top of a tier.

    "Design an Ordered Stream" is the case that motivated this: 82% acceptance,
    13% approval over 4,115 votes, and first in its tier under the old key.
    """
    return (1.0 - approval_weight) * (problem.acceptance_rate / 100.0) + (
        approval_weight * problem.approval
    )


def order_ladder(
    members: list[Problem], *, approval_weight: float = config.APPROVAL_WEIGHT
) -> list[Problem]:
    """Sort a family into study order.

    Easy then Medium then Hard, and within a tier by `rung_quality`. The
    trailing slug key is not cosmetic: without it two problems with identical
    difficulty and quality could order differently between runs, making the
    ladder non-reproducible.
    """
    return sorted(
        members,
        key=lambda p: (p.difficulty_rank, -rung_quality(p, approval_weight=approval_weight), p.slug),
    )


def choose_start(ladder: list[Problem]) -> str | None:
    """Pick the entry point: the most-accepted problem in the easiest tier.

    Not simply the first rung. The easiest tier can contain a problem that is
    simple to state but rarely solved, and acceptance rate is the better proxy
    for "a student can finish this today".
    """
    if not ladder:
        return None
    easiest_rank = ladder[0].difficulty_rank
    tier = [p for p in ladder if p.difficulty_rank == easiest_rank]
    return max(tier, key=lambda p: (rung_quality(p), p.slug)).slug


def build_families(
    communities: list[set[str]],
    problems: list[Problem],
    *,
    min_size: int = config.MIN_FAMILY_SIZE,
) -> list[Family]:
    """Turn raw communities into named, ordered families."""
    by_slug = {p.slug: p for p in problems}
    corpus_tag_freq = Counter(tag for p in problems for tag in p.topics)
    corpus_token_freq: Counter = Counter()
    for problem in problems:
        for token in _title_tokens(problem.title):
            corpus_token_freq[token] += 1

    families: list[Family] = []
    for family_id, community in enumerate(communities):
        members = [by_slug[s] for s in sorted(community) if s in by_slug]
        if len(members) < min_size:
            continue

        ladder = order_ladder(members)
        tags = _distinctive_tags(members, corpus_tag_freq, len(problems))
        token = _distinctive_title_token(members, corpus_token_freq, len(problems))

        parts = ([token] if token else []) + tags[:2]
        # Deduplicate case-insensitively: a word appearing as both a title
        # token and a tag would otherwise read as a stutter.
        seen: set[str] = set()
        unique = []
        for part in parts:
            if part and part.lower() not in seen:
                seen.add(part.lower())
                unique.append(part)
        name = NAME_SEPARATOR.join(unique) if unique else f"Family {family_id}"

        families.append(
            Family(
                family_id=family_id,
                name=name,
                tags=tags,
                members=[p.slug for p in ladder],
                size=len(ladder),
                start_here=choose_start(ladder),
                difficulty_spread=dict(Counter(p.difficulty for p in ladder)),
            )
        )
    return families


# --- Evaluation --------------------------------------------------------------


def normalised_mutual_information(labels_a: list[str], labels_b: list[str]) -> float:
    """Symmetric NMI between two partitions of the same items.

    This exists to test the project's central claim: that these families are
    not LeetCode's tags rediscovered. If NMI against a tag-derived partition
    were near 1.0, the families would be tags and the claim would be false.

    Implemented here rather than imported from scikit-learn because that adds a
    large dependency to a deployment with a 1GB memory ceiling, for one
    function.

    Returns 0.0 when either partition is degenerate (a single cluster): mutual
    information is then 0 and the normaliser would divide by zero.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("partitions must cover the same items")
    n = len(labels_a)
    if n == 0:
        return 0.0

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    joint = Counter(zip(labels_a, labels_b, strict=True))

    def entropy(counts: Counter) -> float:
        return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)

    h_a, h_b = entropy(count_a), entropy(count_b)
    if h_a <= 0.0 or h_b <= 0.0:
        return 0.0

    mutual = 0.0
    for (a, b), c in joint.items():
        p_ab = c / n
        mutual += p_ab * math.log(p_ab / ((count_a[a] / n) * (count_b[b] / n)))

    return max(0.0, 2.0 * mutual / (h_a + h_b))


def tag_partition_labels(problems: list[Problem]) -> list[str]:
    """Label each problem by its full sorted tag set: what tags alone can say."""
    return ["|".join(sorted(p.topics)) if p.topics else "<untagged>" for p in problems]


def family_partition_labels(families: list[Family], problems: list[Problem]) -> list[str]:
    """Label each problem by its family id, or a unique label if it has none.

    Unassigned problems get a unique label rather than a shared "none" bucket:
    lumping them together would assert they form one cluster, which is a claim
    the clustering never made.
    """
    membership = {slug: str(f.family_id) for f in families for slug in f.members}
    return [membership.get(p.slug, f"<none:{p.slug}>") for p in problems]


def coherence(families: list[Family], problems: list[Problem]) -> dict[str, float]:
    """Share of members that actually carry one of their family's naming tags.

    A family is presented to a student under a name derived from its dominant
    tags. If a member carries none of those tags, the name is a false promise
    for that member -- and because strays tend to be Easy with high acceptance,
    they sort to the top of the ladder where they do the most damage.

    `head_coherence` measures the first three rungs specifically, since that is
    what a student actually reads. It is reported separately because overall
    coherence can look healthy while the visible part of every ladder is wrong.
    """
    by_slug = {p.slug: p for p in problems}
    total = matched = head_total = head_matched = 0

    for family in families:
        naming = set(family.tags)
        if not naming:
            continue
        for position, slug in enumerate(family.members):
            problem = by_slug.get(slug)
            if problem is None:
                continue
            hit = bool(naming & set(problem.topics))
            total += 1
            matched += hit
            if position < 3:
                head_total += 1
                head_matched += hit

    return {
        "coherence": round(matched / total, 4) if total else 0.0,
        "head_coherence": round(head_matched / head_total, 4) if head_total else 0.0,
    }


def tag_independence(families: list[Family], problems: list[Problem]) -> dict[str, float | int]:
    """How much of the family structure is explained by tags, over *assigned* items.

    Computing NMI across the whole corpus is wrong here, and measurably so.
    Roughly half the corpus lands in no family, and those problems receive a
    unique label in the family partition. Many also carry a rare tag
    combination, giving them a near-unique label in the tag partition too. The
    two partitions then "agree" on a large block of mutual singletons, and NMI
    is dominated by that agreement rather than by anything about the families.
    The effect is large: on the link-only graph it reports ~0.75, which would
    suggest the families are mostly tags, when the number is mostly an artefact
    of what was left out.

    Restricting to problems that actually got a family answers the question the
    project needs answered: given the problems we do place, is the placement
    just their tags?
    """
    membership = {slug: str(f.family_id) for f in families for slug in f.members}
    assigned = [p for p in problems if p.slug in membership]
    if not assigned:
        return {"nmi": 0.0, "assigned": 0}

    return {
        "nmi": round(
            normalised_mutual_information(
                [membership[p.slug] for p in assigned],
                tag_partition_labels(assigned),
            ),
            4,
        ),
        "assigned": len(assigned),
    }


def link_related_families(
    families: list[Family],
    problems: list[Problem],
    *,
    limit: int = config.RELATED_FAMILIES,
    min_links: int = 1,
) -> list[Family]:
    """Record, for each family, the families it is most connected to.

    A ladder answers "what should I practise next within this pattern". This
    answers the question after that one: which pattern does this one lead to.
    The graph already knows -- curated links that cross a family boundary are
    a person saying two problems are similar despite the clustering having
    separated them -- and nothing was reading it.

    Only *curated* edges count. Inferred embedding edges were added to attach
    strays, and two families being textually adjacent is not evidence that one
    prepares you for the other; it is mostly evidence that both mention arrays.
    Requiring `min_links` distinct crossings also stops a single stray link
    from asserting a relationship between unrelated patterns.
    """
    membership = {slug: f.family_id for f in families for slug in f.members}
    sizes = {f.family_id: max(f.size, 1) for f in families}

    # Distinct *problem pairs*, not link endpoints. The upstream lists are
    # largely reciprocal, so counting endpoints double-counts every
    # relationship -- the raw tallies came out as 2, 4 and 6 with no odd
    # numbers anywhere, which is what gave it away.
    pairs: set[tuple[str, str]] = set()
    for problem in problems:
        source = membership.get(problem.slug)
        if source is None:
            continue
        for target_slug in problem.similar_slugs:
            target = membership.get(target_slug)
            if target is not None and target != source:
                pairs.add((min(problem.slug, target_slug), max(problem.slug, target_slug)))

    crossings: Counter = Counter()
    for left_slug, right_slug in pairs:
        left, right = membership[left_slug], membership[right_slug]
        crossings[(min(left, right), max(left, right))] += 1

    neighbours: dict[int, list[tuple[float, int, int]]] = {}
    for (left, right), count in crossings.items():
        if count < min_links:
            continue
        # Normalised by family size, because two large families cross by
        # coincidence more often than two small ones. Ranking on the raw count
        # offered "randomly picking elements uniformly from a set" as the next
        # step after monotonic stacks, purely because that family is large.
        strength = count / math.sqrt(sizes.get(left, 1) * sizes.get(right, 1))
        neighbours.setdefault(left, []).append((strength, count, right))
        neighbours.setdefault(right, []).append((strength, count, left))

    linked = []
    for family in families:
        ranked = sorted(
            neighbours.get(family.family_id, []),
            # Strongest first; family_id breaks ties so the order is stable.
            key=lambda item: (-item[0], -item[1], item[2]),
        )
        linked.append(
            replace(family, related=[fid for _strength, _count, fid in ranked[:limit]])
        )
    return linked
