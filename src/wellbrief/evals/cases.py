"""Case files: `evals/cases/<suite>.toml`, read with `tomllib` and validated strictly.

A file holds `suite = "<name>"` (its file stem) and a list of `[[case]]`
tables. Every case has `id`, `category` and optionally `gate` (default true)
and `reason` (required when `gate = false`, not allowed otherwise). The other
keys depend on the category, and each category belongs to one suite (the
arithmetic category to two). A missing required key, an unknown key, a value
of the wrong type or out of range, or a category in the wrong suite is an
error, so a typo can never silently turn a check off.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUITES = ("original", "extended", "brief-precision")

# Value checks per key: a type, or a tuple of allowed values.
_NUMBER = (int, float)
KEY_TYPES: dict[str, Any] = {
    "question": str, "questions": list, "from_cases": list,
    "field": str, "well": str, "td_m": _NUMBER, "spread_rate": _NUMBER,
    "top_k": int, "min_hits": int, "doc_type": ("ddr", "eowr", "incident"),
    "code": str, "formation": str, "pattern": ("G1", "G2", "G3", "G4"),
    "figure": ("total_hours", "total_cost_usd", "avoidable_share"),
    "format": ("pdf", "docx", "csv"),
    "min_precision": _NUMBER, "min_accuracy": _NUMBER, "min_practice_precision": _NUMBER,
    "gold_sql": str, "relevant_sql": str, "precondition_sql": str,
}

# Numbers that must be positive, and targets that are shares in [0, 1].
_POSITIVE = {"td_m", "spread_rate", "top_k", "min_hits"}
_SHARE = {"min_precision", "min_accuracy", "min_practice_precision"}

_BRIEF = {"field", "well", "td_m"}

# category -> (required keys, optional keys), besides id / category / gate / reason.
CATEGORIES: dict[str, tuple[set[str], set[str]]] = {
    # original suite: the prototype harness rules
    "retrieval": ({"question", "field"}, {"gold_sql", "doc_type", "top_k", "min_hits"}),
    "grounding": ({"from_cases"}, {"top_k"}),
    "discovery": ({"field", "code", "formation"}, {"precondition_sql"}),
    "brief-citations": (_BRIEF, set()),
    "brief-mitigations": (_BRIEF, set()),
    # both suites
    "arithmetic": ({"question", "figure", "gold_sql"}, {"spread_rate"}),
    # extended suite (retrieval precision is always measured at 8: no top_k)
    "retrieval-precision": ({"question", "field", "relevant_sql"}, set()),
    "abstention": ({"question", "precondition_sql"}, set()),
    "brief-recall": (_BRIEF | {"pattern"}, {"precondition_sql"}),
    "brief-driver": (_BRIEF | {"pattern"}, {"precondition_sql"}),
    "mitigation-precision": (_BRIEF, set()),
    "mitigation-recall": (_BRIEF | {"pattern"}, {"precondition_sql"}),
    "parser-fidelity": ({"field", "gold_sql"}, set()),
    "format-parity": ({"format", "questions"}, set()),
    # brief-precision suite
    "brief-precision": (_BRIEF | {"min_precision"}, set()),
    "classifier-accuracy": ({"gold_sql", "min_accuracy"}, {"min_practice_precision"}),
}

# The suites each category may appear in.
CATEGORY_SUITES: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(("retrieval", "grounding", "discovery", "brief-citations", "brief-mitigations"),
                    ("original",)),
    "arithmetic": ("original", "extended"),
    **dict.fromkeys(("retrieval-precision", "abstention", "brief-recall", "brief-driver",
                     "mitigation-precision", "mitigation-recall", "parser-fidelity", "format-parity"),
                    ("extended",)),
    **dict.fromkeys(("brief-precision", "classifier-accuracy"), ("brief-precision",)),
}

_COMMON = {"id", "category", "gate", "reason"}


class CaseError(ValueError):
    """A case file that does not follow the format."""


@dataclass(frozen=True)
class Case:
    suite: str
    id: str
    category: str
    gate: bool
    reason: str | None
    params: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.params[key]


def _check_value(where: str, key: str, value: Any) -> None:
    expected = KEY_TYPES[key]
    if isinstance(expected, tuple) and all(isinstance(v, str) for v in expected):
        if value not in expected:
            raise CaseError(f"{where}: {key} must be one of {', '.join(expected)}, not {value!r}")
        return
    if isinstance(value, bool) and expected is not bool:
        raise CaseError(f"{where}: {key} has the wrong type (bool)")
    if not isinstance(value, expected):
        raise CaseError(f"{where}: {key} has the wrong type ({type(value).__name__})")
    if isinstance(value, str) and not value.strip():
        raise CaseError(f"{where}: {key} is empty")
    if isinstance(value, list) and (not value or not all(isinstance(v, str) and v.strip() for v in value)):
        raise CaseError(f"{where}: {key} must be a non-empty list of strings")
    if key in _POSITIVE and not value > 0:
        raise CaseError(f"{where}: {key} must be positive, not {value!r}")
    if key in _SHARE and not 0 <= value <= 1:
        raise CaseError(f"{where}: {key} must be between 0 and 1, not {value!r}")


def parse_case(suite: str, raw: Any, where: str) -> Case:
    if not isinstance(raw, dict):
        raise CaseError(f"{where}: a case must be a table, not {type(raw).__name__}")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise CaseError(f"{where}: every case needs a string id")
    where = f"{where} ({case_id})"
    category = raw.get("category")
    if category not in CATEGORIES:
        raise CaseError(f"{where}: unknown category {category!r}")
    if suite not in CATEGORY_SUITES[category]:
        raise CaseError(f"{where}: category {category} does not belong in the {suite} suite")
    required, optional = CATEGORIES[category]
    unknown = sorted(set(raw) - _COMMON - required - optional)
    if unknown:
        raise CaseError(f"{where}: unknown key(s) for category {category}: {', '.join(unknown)}")
    missing = sorted(required - set(raw))
    if missing:
        raise CaseError(f"{where}: missing key(s) for category {category}: {', '.join(missing)}")
    gate = raw.get("gate", True)
    if not isinstance(gate, bool):
        raise CaseError(f"{where}: gate must be true or false")
    reason = raw.get("reason")
    if not gate and not (isinstance(reason, str) and reason.strip()):
        raise CaseError(f"{where}: a case with gate = false needs a reason")
    if gate and reason is not None:
        raise CaseError(f"{where}: reason is only allowed with gate = false")
    params = {k: v for k, v in raw.items() if k not in _COMMON}
    for key, value in params.items():
        _check_value(where, key, value)
    return Case(suite, case_id, category, gate, reason, params)


def parse_suite(data: dict[str, Any], name: str, where: str) -> list[Case]:
    unknown = sorted(set(data) - {"suite", "case"})
    if unknown:
        raise CaseError(f"{where}: unknown top-level key(s): {', '.join(unknown)}")
    if data.get("suite") != name:
        raise CaseError(f"{where}: suite must be {name!r}")
    raw_cases = data.get("case")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise CaseError(f"{where}: no [[case]] tables")
    cases = [parse_case(name, raw, f"{where} case {i}") for i, raw in enumerate(raw_cases, start=1)]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CaseError(f"{where}: duplicate case id(s): {', '.join(duplicates)}")
    by_id = {c.id: c for c in cases}
    for c in cases:
        for ref in c.get("from_cases", []):
            if by_id.get(ref) is None or by_id[ref].category != "retrieval":
                raise CaseError(f"{where} ({c.id}): from_cases names {ref!r}, "
                                "not a retrieval case of this file")
    return cases


def load_suite(cases_dir: Path, name: str) -> list[Case]:
    if name not in SUITES:
        raise CaseError(f"unknown suite {name!r}; choose from {', '.join(SUITES)}")
    path = cases_dir / f"{name}.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CaseError(f"{path.name}: {exc}") from exc
    return parse_suite(data, name, path.name)


def default_dir() -> Path:
    """`evals/cases` of the source checkout this package runs from, else of the working directory."""
    checkout = Path(__file__).resolve().parents[3] / "evals" / "cases"
    return checkout if checkout.is_dir() else Path("evals") / "cases"
