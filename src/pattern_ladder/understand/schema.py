"""The structured reading of a query, and how it is validated.

The point of parsing a query into a schema is not that the schema is clever --
it has three fields. It is that the reading becomes *visible*. A student who
types a sentence about shrinking windows and is shown "technique:
sliding-window, difficulty: any, mode: ramp" can see immediately when the
system has misread them. A system that is silently wrong is worse than one that
is visibly wrong, because the student blames themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIFFICULTIES = ("Easy", "Medium", "Hard")
MODES = ("ramp", "single")

# Where the reading came from. Surfaced in the UI: a fallback parse is a weaker
# claim than a model parse and should not be presented as though it were the
# same thing.
SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class Intent:
    """A structured reading of what the student asked for."""

    technique: str | None = None
    difficulty: str | None = None
    mode: str = "ramp"
    source: str = SOURCE_NONE
    notes: list[str] = field(default_factory=list)

    def to_search_text(self, query: str) -> str:
        """The string the retrievers actually see.

        The recovered technique is appended rather than replacing the query.
        Replacing it would discard the student's own words, which are often the
        only thing distinguishing two problems inside the same technique; and
        if the parse is wrong, appending degrades the query while replacing
        destroys it.
        """
        if self.technique and self.technique.lower() not in query.lower():
            return f"{query} {self.technique}"
        return query

    def describe(self) -> str:
        """Human-readable summary, shown back to the student."""
        return (
            f"technique: {self.technique or 'any'}  |  "
            f"difficulty: {self.difficulty or 'any'}  |  "
            f"mode: {self.mode}"
        )


def coerce(payload: object, *, source: str) -> Intent:
    """Validate an arbitrary object into an Intent, never raising.

    A model returning something unexpected must degrade to a usable Intent, not
    take the request down. Every field is independently validated: a valid
    technique alongside a nonsense difficulty keeps the technique.
    """
    notes: list[str] = []
    if not isinstance(payload, dict):
        return Intent(source=SOURCE_FALLBACK, notes=["response was not an object"])

    technique = payload.get("technique")
    if technique is not None and not isinstance(technique, str):
        notes.append("technique was not a string; ignored")
        technique = None
    if isinstance(technique, str):
        technique = technique.strip() or None
        # Models sometimes answer the literal word "any" or "none" rather than
        # omitting the field.
        if technique and technique.lower() in {"any", "none", "null", "unknown"}:
            technique = None

    difficulty = payload.get("difficulty")
    if isinstance(difficulty, str):
        match = {d.lower(): d for d in DIFFICULTIES}.get(difficulty.strip().lower())
        if match is None and difficulty.strip().lower() not in {"any", "", "all"}:
            notes.append(f"unrecognised difficulty {difficulty!r}; ignored")
        difficulty = match
    elif difficulty is not None:
        notes.append("difficulty was not a string; ignored")
        difficulty = None

    mode = payload.get("mode")
    if not isinstance(mode, str) or mode.strip().lower() not in MODES:
        if mode is not None:
            notes.append(f"unrecognised mode {mode!r}; defaulted to ramp")
        mode = "ramp"
    else:
        mode = mode.strip().lower()

    return Intent(
        technique=technique,
        difficulty=difficulty,
        mode=mode,
        source=source,
        notes=notes,
    )
