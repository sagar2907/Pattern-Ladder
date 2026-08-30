"""Tests for grounded result explanations.

The load-bearing property is negative: an explanation must never assert a
relationship that was not established elsewhere in the pipeline. A confident
false explanation of a correct result is worse than no explanation, because it
teaches a student a connection that does not exist.
"""

from __future__ import annotations

from pattern_ladder.explain import MATCH_BOTH, MATCH_DENSE, MATCH_LEXICAL, explain, match_kind
from pattern_ladder.graph.families import Family


def _family(name="Sliding Window / String", size=12) -> Family:
    return Family(
        family_id=1,
        name=name,
        tags=["Sliding Window", "String"],
        members=["a", "b"],
        size=size,
        start_here="a",
    )


class TestMatchKind:
    def test_both_arms(self):
        assert match_kind(True, True) == MATCH_BOTH

    def test_lexical_only(self):
        assert match_kind(True, False) == MATCH_LEXICAL

    def test_dense_only(self):
        assert match_kind(False, True) == MATCH_DENSE


class TestExplain:
    def test_states_the_difficulty_and_acceptance_actually_retrieved(self, problem_factory):
        problem = problem_factory("p", 1, "P", "Medium", 47.3)
        text = explain(problem, kind=MATCH_BOTH, family=None)
        assert "Medium" in text
        assert "47%" in text

    def test_names_the_family_when_there_is_one(self, problem_factory):
        problem = problem_factory("p", 1, "P")
        text = explain(problem, kind=MATCH_BOTH, family=_family())
        assert "Sliding Window / String" in text
        assert "12" in text

    def test_omits_family_language_when_there_is_none(self, problem_factory):
        problem = problem_factory("p", 1, "P")
        text = explain(problem, kind=MATCH_BOTH, family=None)
        assert "family" not in text.lower()

    def test_lexical_and_dense_matches_are_described_differently(self, problem_factory):
        problem = problem_factory("p", 1, "P")
        lexical = explain(problem, kind=MATCH_LEXICAL, family=None)
        dense = explain(problem, kind=MATCH_DENSE, family=None)
        assert lexical != dense

    def test_no_relation_is_claimed_across_different_families(self, problem_factory):
        """The comparison to the anchor is only meaningful inside one family.
        Asserting it across families would invent a relationship."""
        anchor = problem_factory("anchor", 1, "Anchor", "Easy", 80.0)
        problem = problem_factory("p", 2, "P", "Hard", 30.0)
        text = explain(
            problem,
            kind=MATCH_BOTH,
            family=_family(),
            anchor=anchor,
            shared_family_with_anchor=False,
        )
        assert "Anchor" not in text

    def test_a_step_up_is_claimed_only_when_difficulty_actually_increases(self, problem_factory):
        anchor = problem_factory("anchor", 1, "Anchor", "Easy", 80.0)
        harder = problem_factory("p", 2, "P", "Hard", 30.0)
        text = explain(
            harder,
            kind=MATCH_BOTH,
            family=_family(),
            anchor=anchor,
            shared_family_with_anchor=True,
        )
        assert "step up" in text
        assert "Anchor" in text

    def test_a_gentler_entry_is_claimed_when_difficulty_decreases(self, problem_factory):
        anchor = problem_factory("anchor", 1, "Anchor", "Hard", 30.0)
        easier = problem_factory("p", 2, "P", "Easy", 80.0)
        text = explain(
            easier,
            kind=MATCH_BOTH,
            family=_family(),
            anchor=anchor,
            shared_family_with_anchor=True,
        )
        assert "gentler" in text

    def test_equal_difficulty_and_acceptance_claims_no_relation(self, problem_factory):
        """Saying two problems are equally hard is not a reason to solve one
        after the other, so nothing should be asserted."""
        anchor = problem_factory("anchor", 1, "Anchor", "Medium", 50.0)
        peer = problem_factory("p", 2, "P", "Medium", 50.0)
        text = explain(
            peer, kind=MATCH_BOTH, family=_family(), anchor=anchor,
            shared_family_with_anchor=True,
        )
        assert "Anchor" not in text

    def test_a_problem_is_never_compared_to_itself(self, problem_factory):
        problem = problem_factory("p", 1, "P", "Medium", 50.0)
        text = explain(
            problem, kind=MATCH_BOTH, family=_family(), anchor=problem,
            shared_family_with_anchor=True,
        )
        assert text.count("P") >= 0
        assert "step up" not in text
        assert "gentler" not in text
