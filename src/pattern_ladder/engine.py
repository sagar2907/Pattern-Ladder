"""Assembly: load cached artefacts into a ready SearchEngine.

Kept separate from `index.build` so that serving code never imports the build
path, and separate from `retrieval.search` so the engine class stays a pure
data structure that tests can construct from fixtures without touching disk.
"""

from __future__ import annotations

from dataclasses import replace

from . import config
from .graph.families import Family
from .graph.naming import normalise_description
from .index.build import load_all
from .retrieval.search import SearchEngine


def load_engine(paths: config.Paths | None = None) -> SearchEngine:
    problems, lexical, dense, families_raw = load_all(paths)
    families = [Family(**row) for row in families_raw]

    # Descriptions are the only text in the system written by a model, and they
    # are normalised again here rather than trusted from disk. Validation at
    # generation time can only ever be as good as the rules in force on the day
    # the index was built -- one earlier version let a non-breaking hyphen
    # through, which looks identical on screen and behaves differently
    # everywhere else. Re-checking on load means a stored description can never
    # be worse than the current rules, whenever it was written.
    families = [
        family
        if family.description is None
        else replace(family, description=normalise_description(family.description))
        for family in families
    ]
    return SearchEngine(problems, lexical, dense, families)
