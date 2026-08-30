"""Tests for query understanding: schema validation, the offline parser, and
the guarantee that a model failure degrades rather than propagates.

Nothing here makes a network call. `allow_network=False` is the switch that
makes the whole understanding layer testable without a key, and the monkeypatch
tests below cover the failure paths that a live key would otherwise hide.
"""

from __future__ import annotations

import contextlib

from pattern_ladder.understand import fallback
from pattern_ladder.understand.groq_client import _extract_json, understand
from pattern_ladder.understand.schema import SOURCE_MODEL, Intent, coerce

FENCED_JSON = "".join(["```json\n", '{"technique": "a"}', "\n```"])


class TestSchema:
    def test_valid_payload_is_accepted(self):
        intent = coerce(
            {"technique": "sliding window", "difficulty": "Medium", "mode": "single"},
            source=SOURCE_MODEL,
        )
        assert intent.technique == "sliding window"
        assert intent.difficulty == "Medium"
        assert intent.mode == "single"

    def test_a_non_object_response_never_raises(self):
        assert coerce("not an object", source=SOURCE_MODEL).technique is None
        assert coerce(None, source=SOURCE_MODEL).mode == "ramp"
        assert coerce([1, 2, 3], source=SOURCE_MODEL).technique is None

    def test_fields_are_validated_independently(self):
        """A nonsense difficulty must not discard a good technique."""
        intent = coerce(
            {"technique": "binary search", "difficulty": "Impossible"}, source=SOURCE_MODEL
        )
        assert intent.technique == "binary search"
        assert intent.difficulty is None
        assert any("Impossible" in note for note in intent.notes)

    def test_difficulty_is_case_insensitive(self):
        assert coerce({"difficulty": "hARd"}, source=SOURCE_MODEL).difficulty == "Hard"

    def test_the_literal_word_any_is_treated_as_absent(self):
        """Models answer with the word 'any' instead of omitting the field."""
        assert coerce({"technique": "any"}, source=SOURCE_MODEL).technique is None
        assert coerce({"difficulty": "any"}, source=SOURCE_MODEL).difficulty is None

    def test_unknown_mode_defaults_to_ramp(self):
        intent = coerce({"mode": "sideways"}, source=SOURCE_MODEL)
        assert intent.mode == "ramp"
        assert intent.notes

    def test_non_string_fields_are_rejected_with_a_note(self):
        intent = coerce({"technique": 42, "difficulty": []}, source=SOURCE_MODEL)
        assert intent.technique is None
        assert intent.difficulty is None
        assert len(intent.notes) == 2

    def test_whitespace_only_technique_is_absent(self):
        assert coerce({"technique": "   "}, source=SOURCE_MODEL).technique is None


class TestSearchText:
    def test_technique_is_appended_not_substituted(self):
        """Replacing the query would discard the student's own words, which are
        often the only thing separating two problems inside one technique. It
        also makes a wrong parse destructive rather than merely unhelpful."""
        intent = Intent(technique="sliding window")
        text = intent.to_search_text("shrink from the left")
        assert "shrink from the left" in text
        assert "sliding window" in text

    def test_technique_already_present_is_not_duplicated(self):
        intent = Intent(technique="sliding window")
        assert intent.to_search_text("sliding window problems").count("sliding window") == 1

    def test_no_technique_leaves_the_query_untouched(self):
        assert Intent().to_search_text("anything") == "anything"


class TestFallbackParser:
    def test_oblique_phrasing_recovers_the_technique(self):
        """The case the project exists for: the student describes the mechanic
        without naming it, so no keyword index can reach the right problems."""
        assert fallback.parse("I shrink a window from the left").technique == "sliding window"

    def test_longer_cues_win_over_shorter_ones(self):
        """'binary search tree' must not be read as 'binary search'."""
        assert fallback.parse("traverse a binary search tree").technique == "tree"

    def test_punctuation_and_hyphens_do_not_block_a_match(self):
        assert fallback.parse("window-from-the-left!").technique == "sliding window"

    def test_difficulty_cues_are_recognised(self):
        assert fallback.parse("easy warm up problems").difficulty == "Easy"
        assert fallback.parse("something challenging").difficulty == "Hard"

    def test_single_and_ramp_modes(self):
        assert fallback.parse("just one problem on stacks").mode == "single"
        assert fallback.parse("a ladder of stack problems").mode == "ramp"

    def test_default_mode_is_ramp(self):
        assert fallback.parse("stacks").mode == "ramp"

    def test_unrecognised_query_returns_a_usable_intent_with_a_note(self):
        intent = fallback.parse("qwertyuiop zxcvbnm")
        assert intent.technique is None
        assert intent.mode == "ramp"
        assert intent.notes

    def test_parsing_is_deterministic(self):
        assert fallback.parse("shrink the window") == fallback.parse("shrink the window")


class TestJSONExtraction:
    def test_plain_json(self):
        assert _extract_json('{"technique": "a"}') == {"technique": "a"}

    def test_json_inside_a_fenced_block(self):
        """Even with a JSON response format enforced, models emit fences."""
        assert _extract_json(FENCED_JSON) == {"technique": "a"}

    def test_json_after_a_leading_sentence(self):
        assert _extract_json('Sure! {"mode": "ramp"}') == {"mode": "ramp"}

    def test_prose_with_no_json_returns_none(self):
        assert _extract_json("I cannot help with that.") is None

    def test_empty_returns_none(self):
        assert _extract_json("") is None
        assert _extract_json("   ") is None


class TestOfflineDegradation:
    def test_offline_mode_uses_the_rule_parser_and_says_so(self):
        intent = understand("shrink a window from the left", allow_network=False)
        assert intent.technique == "sliding window"
        assert intent.source == "fallback"
        assert any("rule-based" in note for note in intent.notes)

    def test_offline_mode_never_raises_on_junk(self):
        assert understand("", allow_network=False) is not None
        assert understand("!!!", allow_network=False).mode == "ramp"

    def test_a_failing_model_call_degrades_to_the_rule_parser(self, monkeypatch):
        """Contract: understand() always returns an Intent. Network failure,
        rate limiting and malformed output must never reach the request."""
        monkeypatch.setattr("pattern_ladder.understand.groq_client.available", lambda: True)

        def explode(*_args, **_kwargs):
            raise ConnectionError("network is down")

        monkeypatch.setattr("pattern_ladder.understand.groq_client.call_model", explode)
        intent = understand("shrink a window from the left")
        assert intent.technique == "sliding window"
        assert any("model call failed" in note for note in intent.notes)

    def test_unparseable_model_output_degrades_to_the_rule_parser(self, monkeypatch):
        monkeypatch.setattr("pattern_ladder.understand.groq_client.available", lambda: True)
        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client.call_model",
            lambda *_a, **_k: ("I am a language model and cannot comply.", {}),
        )
        intent = understand("next greater element")
        assert intent.technique == "monotonic stack"
        assert any("unparseable" in note for note in intent.notes)

    def test_model_parse_missing_a_technique_borrows_the_rule_one(self, monkeypatch):
        """A model parse with no technique is weaker than the rule parse, which
        at least matched a known phrasing. The other model fields are kept."""
        monkeypatch.setattr("pattern_ladder.understand.groq_client.available", lambda: True)
        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client.call_model",
            lambda *_a, **_k: ('{"technique": null, "difficulty": "Hard"}', {}),
        )
        intent = understand("shrink a window from the left")
        assert intent.technique == "sliding window"
        assert intent.difficulty == "Hard"


class TestLiveCallParameters:
    """Guards on the request itself, verified without making one.

    These pin defects that only appeared against the live API and that the
    offline suite could not otherwise catch -- because every one of them
    degrades silently to the rule-based parser, which works well enough that
    nothing looks broken.
    """

    def test_token_budget_covers_reasoning_tokens(self):
        """Regression: max_tokens=200 failed every single live call.

        The configured model is a reasoning model. It emits several hundred
        tokens of reasoning before the JSON answer, and those count against
        max_tokens. At 200 the object was cut off mid-generation and the API
        returned 400 json_validate_failed on every request -- so the model path
        never once succeeded, while the system looked healthy because it fell
        back to the offline parser each time.
        """
        from pattern_ladder import config

        assert config.GROQ_MAX_OUTPUT_TOKENS >= 400

    def test_request_asks_for_low_reasoning_effort(self, monkeypatch):
        """Reasoning tokens are billed and rate-limited like any other.

        Measured live: "low" costs 295 tokens per query against 488 at the
        default, runs faster, and extracted a technique the default missed.
        Since the free tier's binding constraint is tokens per minute, this
        parameter is what decides throughput.
        """
        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                raise RuntimeError("stop before any network call")

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        monkeypatch.setattr(
            "pattern_ladder.understand.groq_client._client", lambda: _Client()
        )
        from pattern_ladder.understand.groq_client import call_model

        # The stub raises to guarantee nothing reaches the network; the
        # request parameters have already been captured by then.
        with contextlib.suppress(RuntimeError):
            call_model("anything")

        assert captured["reasoning_effort"] == "low"
        assert captured["temperature"] == 0.0
        assert captured["response_format"] == {"type": "json_object"}
        assert captured["max_tokens"] >= 400

    def test_prompt_states_that_ramp_is_the_default(self):
        """Regression: the model answered "single" to almost everything.

        The original wording offered both modes symmetrically, and the model
        read a description of repeated failure as a request for one problem --
        2 of 5 correct on a hand-checked set. Naming the default explicitly
        took it to 5 of 5.
        """
        from pattern_ladder.understand.groq_client import SYSTEM_PROMPT

        assert "Default to" in SYSTEM_PROMPT
        assert "ramp" in SYSTEM_PROMPT
