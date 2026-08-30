"""A small on-disk cache for query readings.

Three problems, one mechanism.

**Reproducibility.** Temperature 0 is greedy decoding, not a guarantee: of the
twenty evaluation queries repeated three times each, nineteen came back
bit-identical and one did not. A student who asks the same question twice and
is shown two different readings has no way to tell which one the system
"really" thinks, and the whole point of showing the reading is that it can be
checked. A cache makes the answer stable by construction rather than by hope.

**Cost.** The free tier's binding constraint is tokens per minute, and a repeat
query costs 372 of them for an answer already known.

**Latency.** A cached reading is a dictionary lookup instead of a ~0.9s round
trip, which is most of the non-reranking time in a query.

The cache key covers the model *and* the system prompt, not just the query.
That is what makes it safe: editing the prompt -- which is the one change that
should invalidate every stored reading -- produces different keys, so stale
entries become unreachable instead of being silently served. There is no
expiry, because nothing else about a reading goes out of date.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from .. import config
from .schema import Intent

# Beyond this many entries the file is rewritten from scratch keeping the most
# recent half. A study tool sees a few thousand distinct queries at most, so
# this is a guard against unbounded growth rather than a working limit.
MAX_ENTRIES = 5000


def _key(query: str, *, model: str, prompt: str) -> str:
    # The query is stripped and lowercased so that trivial differences in how a
    # student types the same question share an entry; everything semantic about
    # the request is in the model and prompt, which are hashed alongside.
    material = "\x00".join([query.strip().lower(), model, prompt])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class ParseCache:
    """Query reading cache, backed by a single JSON file.

    Reads and writes are individually fault-tolerant: a corrupt or unwritable
    cache degrades to no caching at all, never to a failed query. This sits in
    front of a component whose entire contract is that it does not fail.
    """

    def __init__(self, path: Path | None = None, *, enabled: bool = True) -> None:
        self.path = path or (config.ARTIFACTS_DIR / "parse_cache.json")
        self.enabled = enabled
        self._entries: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self._entries is not None:
            return self._entries
        self._entries = {}
        if self.enabled and self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._entries = {
                        k: v for k, v in loaded.items() if isinstance(v, dict)
                    }
            except (OSError, ValueError):
                # A damaged cache is not worth a failed query, or a traceback.
                self._entries = {}
        return self._entries

    def get(self, query: str, *, model: str, prompt: str) -> Intent | None:
        if not self.enabled:
            return None
        entry = self._load().get(_key(query, model=model, prompt=prompt))
        if not entry:
            return None
        try:
            return Intent(
                technique=entry.get("technique"),
                difficulty=entry.get("difficulty"),
                mode=entry.get("mode", "ramp"),
                source=entry.get("source", "model"),
                # The note is added on read rather than stored, so that a cached
                # reading is visibly cached in the interface. A student looking
                # at a stale answer should be able to see that it is one.
                notes=[*entry.get("notes", []), "reading reused from cache"],
            )
        except (TypeError, ValueError):
            return None

    def put(self, query: str, intent: Intent, *, model: str, prompt: str) -> None:
        if not self.enabled:
            return
        entries = self._load()
        entries[_key(query, model=model, prompt=prompt)] = {
            "technique": intent.technique,
            "difficulty": intent.difficulty,
            "mode": intent.mode,
            "source": intent.source,
            "notes": list(intent.notes),
        }
        if len(entries) > MAX_ENTRIES:
            # dicts preserve insertion order, so the tail is the most recent.
            keep = list(entries.items())[-(MAX_ENTRIES // 2) :]
            entries = dict(keep)
            self._entries = entries
        self._flush(entries)

    def _flush(self, entries: dict[str, dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a sibling and renamed, so a process interrupted
            # mid-write cannot leave a half-written file that the next run
            # treats as a corrupt cache and discards wholesale.
            tmp = self.path.with_suffix(".json.partial")
            tmp.write_text(json.dumps(entries), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def clear(self) -> None:
        self._entries = {}
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)


_DEFAULT: ParseCache | None = None


def default_cache() -> ParseCache:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ParseCache(enabled=config.PARSE_CACHE_ENABLED)
    return _DEFAULT
