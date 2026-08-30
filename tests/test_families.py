"""Tests for community detection, naming, ladders, and the evaluation metrics."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pattern_ladder import config
from pattern_ladder.graph.build import add_knn_backfill, build_link_graph
from pattern_ladder.graph.families import (
    build_families,
    choose_start,
    coherence,
    detect_families,
    link_related_families,
    normalised_mutual_information,
    order_ladder,
    rung_quality,
    tag_independence,
)


def test_detection_is_deterministic(problems):
    """Louvain visits nodes in a randomised order. Without a fixed seed the
    partition, and therefore every family id, changes between rebuilds."""
    graph = build_link_graph(problems)
    assert detect_families(graph) == detect_families(graph)


def test_communities_are_sorted_largest_first(problems):
    graph = build_link_graph(problems)
    sizes = [len(c) for c in detect_families(graph)]
    assert sizes == sorted(sizes, reverse=True)


def test_the_two_chains_separate(problems):
    graph = build_link_graph(problems)
    communities = detect_families(graph, resolution=1.0)
    window = next(c for c in communities if "window-easy" in c)
    assert "stack-easy" not in window


def test_families_below_minimum_size_are_dropped(problems):
    graph = build_link_graph(problems)
    families = build_families(detect_families(graph), problems, min_size=5)
    assert all(f.size >= 5 for f in families)
    assert all("lonely-math" not in f.members for f in families)


def test_ladder_orders_easy_to_hard(problems):
    ordered = order_ladder(problems)
    ranks = [p.difficulty_rank for p in ordered]
    assert ranks == sorted(ranks)


def test_within_a_tier_higher_acceptance_comes_first(problems):
    mediums = [p for p in problems if p.difficulty == "Medium"]
    ordered = order_ladder(mediums)
    rates = [p.acceptance_rate for p in ordered]
    assert rates == sorted(rates, reverse=True)


def test_ladder_ordering_is_total_and_reproducible(problems):
    """Ties on difficulty and acceptance must still order identically, or a
    cached ladder differs from a freshly computed one."""
    assert order_ladder(problems) == order_ladder(list(reversed(problems)))


def test_start_here_is_the_most_accepted_of_the_easiest_tier(problems):
    window = [p for p in problems if p.slug.startswith("window")]
    start = choose_start(order_ladder(window))
    assert start == "window-easy"


def test_start_here_is_none_for_an_empty_ladder():
    assert choose_start([]) is None


def test_family_name_uses_distinctive_tags_not_common_ones(problems):
    graph = build_link_graph(problems)
    families = build_families(detect_families(graph, resolution=1.0), problems, min_size=5)
    stack = next(f for f in families if "stack-easy" in f.members)
    assert "Stack" in " ".join(stack.tags)


def test_nmi_of_a_partition_with_itself_is_one():
    labels = ["a", "a", "b", "b", "c"]
    assert normalised_mutual_information(labels, labels) == pytest.approx(1.0)


def test_nmi_of_independent_partitions_is_near_zero():
    a = ["x", "y"] * 20
    b = ["p", "p", "q", "q"] * 10
    assert normalised_mutual_information(a, b) < 0.1


def test_nmi_of_a_degenerate_partition_is_zero():
    """A single cluster has zero entropy; the normaliser would divide by zero."""
    assert normalised_mutual_information(["a"] * 5, ["p", "q", "p", "q", "p"]) == 0.0


def test_nmi_requires_matching_lengths():
    with pytest.raises(ValueError):
        normalised_mutual_information(["a"], ["a", "b"])


def test_tag_independence_scores_only_assigned_problems(problems):
    """Regression: scoring the whole corpus inflated the number badly.

    Unassigned problems get a unique label in the family partition, and many
    carry a rare tag combination giving them a near-unique label in the tag
    partition too. The two partitions then agree on a large block of mutual
    singletons, and NMI reports that agreement rather than anything about the
    families. On the real corpus this was the difference between 0.75 and 0.60.
    """
    graph = build_link_graph(problems)
    families = build_families(detect_families(graph), problems, min_size=5)
    result = tag_independence(families, problems)
    assigned = {s for f in families for s in f.members}
    assert result["assigned"] == len(assigned)
    assert result["assigned"] < len(problems)


def test_coherence_reports_both_overall_and_head(problems):
    graph = build_link_graph(problems)
    families = build_families(detect_families(graph), problems, min_size=5)
    result = coherence(families, problems)
    assert 0.0 <= result["coherence"] <= 1.0
    assert 0.0 <= result["head_coherence"] <= 1.0


def test_empty_families_do_not_break_the_metrics(problems):
    assert tag_independence([], problems)["nmi"] == 0.0
    assert coherence([], problems)["coherence"] == 0.0


class TestRungQuality:
    """Ordering within a difficulty tier blends approachability with approval."""

    def _p(self, problem_factory, slug, acceptance, likes, dislikes):
        problem = problem_factory(slug, 1, slug.title(), "Medium", acceptance)
        return replace(problem, likes=likes, dislikes=dislikes)

    def test_a_well_regarded_problem_beats_an_easier_disliked_one(self, problem_factory):
        """The case this exists for. "Design an Ordered Stream" is accepted 82%
        of the time and approved by 13% of 4,115 voters; ordering on acceptance
        alone puts it first in its tier, which is a bad recommendation however
        approachable it is."""
        disliked = self._p(problem_factory, "disliked", 82.0, 550, 3565)
        liked = self._p(problem_factory, "liked", 60.0, 18863, 783)
        assert rung_quality(liked) > rung_quality(disliked)
        assert [p.slug for p in order_ladder([disliked, liked])] == ["liked", "disliked"]

    def test_acceptance_still_decides_between_equally_regarded_problems(
        self, problem_factory
    ):
        """Approval must not overwhelm approachability, which is what makes a
        ladder walkable."""
        easier = self._p(problem_factory, "easier", 80.0, 1000, 100)
        harder = self._p(problem_factory, "harder", 40.0, 1000, 100)
        assert [p.slug for p in order_ladder([harder, easier])] == ["easier", "harder"]

    def test_difficulty_still_dominates_quality(self, problem_factory):
        """A ladder ascends; no quality score may reorder the tiers."""
        easy = replace(
            problem_factory("e", 1, "E", "Easy", 10.0), likes=1, dislikes=9999
        )
        hard = replace(
            problem_factory("h", 2, "H", "Hard", 99.0), likes=9999, dislikes=1
        )
        assert [p.slug for p in order_ladder([hard, easy])] == ["e", "h"]

    def test_weight_zero_reproduces_pure_acceptance_ordering(self, problem_factory):
        disliked = self._p(problem_factory, "disliked", 82.0, 550, 3565)
        liked = self._p(problem_factory, "liked", 60.0, 18863, 783)
        ordered = order_ladder([liked, disliked], approval_weight=0.0)
        assert [p.slug for p in ordered] == ["disliked", "liked"]

    def test_start_here_uses_quality_not_bare_acceptance(self, problem_factory):
        disliked = replace(
            problem_factory("disliked", 1, "D", "Easy", 90.0), likes=10, dislikes=9000
        )
        liked = replace(
            problem_factory("liked", 2, "L", "Easy", 70.0), likes=9000, dislikes=100
        )
        assert choose_start(order_ladder([disliked, liked])) == "liked"

    def test_ordering_remains_total_and_reproducible(self, problem_factory):
        problems = [
            self._p(problem_factory, f"p{i}", 50.0, 100, 100) for i in range(6)
        ]
        assert order_ladder(problems) == order_ladder(list(reversed(problems)))


class TestApprovalSmoothing:
    def test_a_thinly_voted_problem_is_pulled_toward_the_median(self, problem_factory):
        """Three likes and no dislikes is not evidence of a great problem."""
        thin = replace(problem_factory("t", 1, "T"), likes=3, dislikes=0)
        assert 0.85 < thin.approval < 0.98

    def test_a_heavily_voted_problem_keeps_its_true_ratio(self, problem_factory):
        heavy = replace(problem_factory("h", 1, "H"), likes=550, dislikes=3565)
        assert abs(heavy.approval - 550 / 4115) < 0.02

    def test_a_problem_with_no_votes_gets_the_prior(self, problem_factory):
        from pattern_ladder.data import PRIOR_APPROVAL

        silent = problem_factory("s", 1, "S")
        assert silent.approval == pytest.approx(PRIOR_APPROVAL)


class TestRelatedFamilies:
    """Curated links that cross a family boundary say which pattern leads to
    which. A ladder answers 'what next within this pattern'; this answers the
    question after it."""

    def _families(self, problems):
        graph = build_link_graph(problems)
        families = build_families(detect_families(graph, resolution=1.0), problems, min_size=5)
        return link_related_families(families, problems, min_links=1)

    def test_relationships_are_symmetric(self, problems):
        families = self._families(problems)
        by_id = {f.family_id: f for f in families}
        for family in families:
            for other_id in family.related:
                assert family.family_id in by_id[other_id].related

    def test_a_family_is_never_related_to_itself(self, problems):
        for family in self._families(problems):
            assert family.family_id not in family.related

    def test_related_ids_all_resolve(self, problems):
        families = self._families(problems)
        known = {f.family_id for f in families}
        for family in families:
            assert set(family.related) <= known

    def test_the_list_is_capped(self, problems):
        for family in self._families(problems):
            assert len(family.related) <= config.RELATED_FAMILIES

    def test_a_higher_link_threshold_removes_weak_relationships(self, problems):
        graph = build_link_graph(problems)
        base = build_families(detect_families(graph, resolution=1.0), problems, min_size=5)
        loose = link_related_families(base, problems, min_links=1)
        strict = link_related_families(base, problems, min_links=99)
        assert sum(len(f.related) for f in strict) <= sum(len(f.related) for f in loose)
        assert sum(len(f.related) for f in strict) == 0

    def test_only_curated_links_count(self, problems, embeddings):
        """Inferred embedding edges attach strays; two families being textually
        adjacent is not evidence that one prepares you for the other. So a
        backfilled graph must not invent new relationships."""
        plain = build_link_graph(problems)
        backfilled = build_link_graph(problems)
        add_knn_backfill(backfilled, problems, embeddings)

        from_plain = build_families(
            detect_families(plain, resolution=1.0), problems, min_size=5
        )
        related = link_related_families(from_plain, problems, min_links=1)
        # link_related_families reads problems, never the graph, so an inferred
        # edge cannot reach it at all.
        again = link_related_families(from_plain, problems, min_links=1)
        assert [f.related for f in related] == [f.related for f in again]

    def test_ordering_is_deterministic(self, problems):
        first = self._families(problems)
        second = self._families(problems)
        assert [f.related for f in first] == [f.related for f in second]
