"""Tests for the query-reading cache.

The cache exists to make a repeated question give a repeated answer, which
greedy decoding alone does not guarantee. It sits in front of a component whose
whole contract is that it never fails, so most of these tests are about the
cache degrading rather than raising.
"""

from __future__ import annotations

from pattern_ladder.understand.cache import MAX_ENTRIES, ParseCache, _key
from pattern_ladder.understand.schema import SOURCE_MODEL, Intent

MODEL = "some/model"
PROMPT = "a system prompt"


def _cache(tmp_path, **kwargs) -> ParseCache:
    return ParseCache(tmp_path / "cache.json", **kwargs)


def _intent(technique: str = "sliding window") -> Intent:
    return Intent(technique=technique, difficulty="Medium", mode="ramp", source=SOURCE_MODEL)


class TestRoundTrip:
    def test_a_stored_reading_comes_back(self, tmp_path):
        cache = _cache(tmp_path)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        got = cache.get("a query", model=MODEL, prompt=PROMPT)
        assert got is not None
        assert got.technique == "sliding window"
        assert got.difficulty == "Medium"
        assert got.mode == "ramp"

    def test_a_miss_returns_none(self, tmp_path):
        assert _cache(tmp_path).get("never seen", model=MODEL, prompt=PROMPT) is None

    def test_a_cached_reading_says_so(self, tmp_path):
        """A student looking at a stored answer should be able to tell."""
        cache = _cache(tmp_path)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        got = cache.get("a query", model=MODEL, prompt=PROMPT)
        assert any("cache" in note for note in got.notes)

    def test_entries_survive_a_new_instance(self, tmp_path):
        _cache(tmp_path).put("a query", _intent(), model=MODEL, prompt=PROMPT)
        assert _cache(tmp_path).get("a query", model=MODEL, prompt=PROMPT) is not None


class TestKeying:
    def test_whitespace_and_case_do_not_split_an_entry(self, tmp_path):
        cache = _cache(tmp_path)
        cache.put("Sliding Window ", _intent(), model=MODEL, prompt=PROMPT)
        assert cache.get("sliding window", model=MODEL, prompt=PROMPT) is not None

    def test_a_changed_prompt_orphans_the_entry(self, tmp_path):
        """The one change that must invalidate every stored reading. Editing the
        prompt changes what the model is being asked, so a reading produced
        under the old wording is not an answer to the new question."""
        cache = _cache(tmp_path)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        assert cache.get("a query", model=MODEL, prompt="a different prompt") is None

    def test_a_changed_model_orphans_the_entry(self, tmp_path):
        cache = _cache(tmp_path)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        assert cache.get("a query", model="other/model", prompt=PROMPT) is None

    def test_keys_are_stable_between_calls(self):
        first = _key("a query", model=MODEL, prompt=PROMPT)
        assert first == _key("a query", model=MODEL, prompt=PROMPT)


class TestFailureModes:
    def test_a_corrupt_file_degrades_to_no_cache(self, tmp_path):
        """A damaged cache must not raise into a query."""
        path = tmp_path / "cache.json"
        path.write_text("this is not json at all", encoding="utf-8")
        cache = ParseCache(path)
        assert cache.get("a query", model=MODEL, prompt=PROMPT) is None
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        assert cache.get("a query", model=MODEL, prompt=PROMPT) is not None

    def test_a_file_holding_the_wrong_shape_is_ignored(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('["not", "a", "mapping"]', encoding="utf-8")
        assert ParseCache(path).get("a query", model=MODEL, prompt=PROMPT) is None

    def test_a_malformed_entry_is_ignored(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text('{"abc": "not an object"}', encoding="utf-8")
        assert ParseCache(path).get("a query", model=MODEL, prompt=PROMPT) is None

    def test_disabling_the_cache_stores_and_returns_nothing(self, tmp_path):
        cache = _cache(tmp_path, enabled=False)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        assert cache.get("a query", model=MODEL, prompt=PROMPT) is None
        assert not (tmp_path / "cache.json").exists()

    def test_clear_removes_everything(self, tmp_path):
        cache = _cache(tmp_path)
        cache.put("a query", _intent(), model=MODEL, prompt=PROMPT)
        cache.clear()
        assert cache.get("a query", model=MODEL, prompt=PROMPT) is None


def test_the_cache_is_bounded(tmp_path):
    """Unbounded growth in a file rewritten on every write is a slow leak."""
    cache = _cache(tmp_path)
    for i in range(MAX_ENTRIES + 50):
        cache.put(f"query {i}", _intent(), model=MODEL, prompt=PROMPT)
    assert len(cache._load()) <= MAX_ENTRIES
    # The most recent write must survive the trim.
    assert cache.get(f"query {MAX_ENTRIES + 49}", model=MODEL, prompt=PROMPT) is not None


def test_understand_prefers_the_cache_over_a_call(monkeypatch, tmp_path):
    """The point of the cache: a second identical question makes no call."""
    calls = []

    monkeypatch.setattr("pattern_ladder.understand.groq_client.available", lambda: True)

    def counting(*_args, **_kwargs):
        calls.append(1)
        return ('{"technique": "monotonic stack", "mode": "ramp"}', {})

    monkeypatch.setattr("pattern_ladder.understand.groq_client.call_model", counting)
    from pattern_ladder.understand.groq_client import understand

    cache = _cache(tmp_path)
    first = understand("next greater element", cache=cache)
    second = understand("next greater element", cache=cache)

    assert len(calls) == 1
    assert first.technique == second.technique == "monotonic stack"


def test_a_failed_call_is_not_cached(monkeypatch, tmp_path):
    """Caching a fallback reading would make one bad minute permanent."""
    monkeypatch.setattr("pattern_ladder.understand.groq_client.available", lambda: True)

    def explode(*_args, **_kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr("pattern_ladder.understand.groq_client.call_model", explode)
    from pattern_ladder.understand.groq_client import understand

    cache = _cache(tmp_path)
    understand("shrink a window from the left", cache=cache)
    assert cache.get("shrink a window from the left", model="openai/gpt-oss-20b",
                     prompt="") is None
    assert cache._load() == {}
