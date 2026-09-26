"""Accuracy harness.

The harness turns the quality of retrieval, arithmetic, grounding and pattern
discovery into a pass count that can be compared from one change to the next.
The gold answers are not hand written strings; they are computed from the NPT
ledger, and the system under test has to reach the same conclusion through
retrieval and narration. If the two diverge, the run fails and says which case
failed.

Four things are measured:

  retrieval   does the top-k contain a document from the right well and the
              right failure class
  arithmetic  do the figures in the structured payload match a direct query
  grounding   is every citation a real document, quoted verbatim
  discovery   does the risk brief rediscover the dominant failure mode of each
              field without being told what it is
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Callable

from .analytics import find_patterns, rollup
from .config import DEFAULT_SPREAD_RATE_USD_PER_DAY
from .llm import Narrator, OfflineNarrator
from .qa import ask
from .riskbrief import build_brief, verify_brief
from .search import Searcher
from .store import Store


@dataclass
class Case:
    case_id: str
    question: str
    expect_code: str | None = None
    expect_field: str | None = None
    expect_doc_type: str | None = None
    min_hits_from_affected: int = 1


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    detail: str
    metrics: dict[str, Any] = dc_field(default_factory=dict)


RETRIEVAL_CASES = [
    Case("orrindale-stuck", "What caused stuck pipe on Orrindale wells in the 17 1/2\" section?",
         expect_code="STUCK_PIPE", expect_field="Orrindale"),
    Case("orrindale-salt", "Show wellbore instability in Keldra Salt on Orrindale",
         expect_code="WELLBORE_INSTABILITY", expect_field="Orrindale"),
    Case("vessra-losses", "Total losses entering Vessra Carbonate on Vessra South",
         expect_code="LOST_CIRCULATION", expect_field="Vessra South"),
    Case("mwd-failure", "MWD tool failures in the 12 1/4\" section on Orrindale",
         expect_code="DOWNHOLE_TOOL_FAILURE", expect_field="Orrindale"),
    Case("rig-repair", "Mud pump problems on Vessra South",
         expect_code="RIG_REPAIR", expect_field="Vessra South"),
    Case("lessons", "What do the end of well reports recommend for Keldra Salt on Orrindale?",
         expect_field="Orrindale", expect_doc_type="eowr"),
    Case("cement", "Cementing problems on Orrindale", expect_code="CEMENT_ISSUE", expect_field="Orrindale"),
    Case("weather", "How much time was lost to weather on Vessra South?",
         expect_code="WAIT_ON_WEATHER", expect_field="Vessra South"),
]

# The dominant failure each field's history should surface on its own.
DISCOVERY_EXPECTATIONS = {
    "Orrindale": ("STUCK_PIPE", "Keldra Salt"),
    "Vessra South": ("LOST_CIRCULATION", "Vessra Carbonate"),
}


def _run_retrieval_case(case: Case, store: Store, searcher: Searcher, narrator: Narrator) -> CaseResult:
    answer = ask(case.question, store, searcher, narrator=narrator, top_k=8)
    docs = [c.doc_id for c in answer.citations]
    if not docs:
        return CaseResult(case.case_id, False, "no documents retrieved")

    if case.expect_field:
        wrong_field = [
            c.doc_id for c in answer.citations
            if (d := store.get_document(c.doc_id)) and d.field_name != case.expect_field
        ]
        if wrong_field:
            return CaseResult(
                case.case_id, False,
                f"retrieved documents from another field: {wrong_field[:3]}",
                {"retrieved": len(docs)},
            )

    if case.expect_code:
        wells_with_code = {
            e.well for e in store.npt(field_name=case.expect_field, code=case.expect_code)
        }
        if not wells_with_code:
            return CaseResult(case.case_id, False, f"no ground truth for {case.expect_code}")
        matched = sum(
            1 for c in answer.citations
            if (d := store.get_document(c.doc_id)) and d.well in wells_with_code
        )
        if matched < case.min_hits_from_affected:
            return CaseResult(
                case.case_id, False,
                f"no retrieved document belongs to a well that had {case.expect_code}",
                {"retrieved": len(docs), "matched": matched},
            )

    if case.expect_doc_type:
        types = {
            d.doc_type for c in answer.citations if (d := store.get_document(c.doc_id))
        }
        if case.expect_doc_type not in types:
            return CaseResult(
                case.case_id, False,
                f"expected a {case.expect_doc_type} in the top results, got {sorted(types)}",
            )

    if answer.structured.get("citation_warnings"):
        return CaseResult(
            case.case_id, False,
            f"citation warnings: {answer.structured['citation_warnings'][:2]}",
        )

    return CaseResult(case.case_id, True, "ok", {"retrieved": len(docs)})


def _check_arithmetic(store: Store, searcher: Searcher, narrator: Narrator) -> list[CaseResult]:
    """The structured payload has to agree with a direct query on the ledger."""
    out: list[CaseResult] = []
    for field_name, code in (("Orrindale", "STUCK_PIPE"), ("Vessra South", "LOST_CIRCULATION")):
        truth = rollup(store.npt(field_name=field_name, code=code))
        answer = ask(
            f"How many hours of {code.replace('_', ' ').lower()} on {field_name}?",
            store, searcher, narrator=narrator,
        )
        got = (answer.structured.get("stats") or {}).get("total_hours")
        ok = got is not None and abs(got - truth["total_hours"]) < 0.05
        out.append(CaseResult(
            f"arithmetic-{field_name.replace(' ', '-').lower()}-{code.lower()}",
            ok,
            "ok" if ok else f"expected {truth['total_hours']} h, payload reported {got}",
            {"expected_hours": truth["total_hours"], "reported_hours": got},
        ))
    return out


def _check_grounding(store: Store, searcher: Searcher, narrator: Narrator) -> CaseResult:
    problems: list[str] = []
    checked = 0
    for case in RETRIEVAL_CASES:
        answer = ask(case.question, store, searcher, narrator=narrator, top_k=6)
        for cite in answer.citations:
            checked += 1
            doc = store.get_document(cite.doc_id)
            if doc is None:
                problems.append(f"{case.case_id}: {cite.doc_id} does not exist")
            elif " ".join(cite.quote.split()) not in " ".join(doc.text.split()):
                problems.append(f"{case.case_id}: quote not verbatim in {cite.doc_id}")
    return CaseResult(
        "grounding", not problems,
        "ok" if not problems else "; ".join(problems[:3]),
        {"citations_checked": checked, "problems": len(problems)},
    )


def _check_discovery(store: Store, narrator: Narrator) -> list[CaseResult]:
    out: list[CaseResult] = []
    for field_name, (code, formation) in DISCOVERY_EXPECTATIONS.items():
        patterns = find_patterns(store, field_name)
        if not patterns:
            out.append(CaseResult(f"discovery-{field_name}", False, "no patterns found"))
            continue
        top = patterns[0]
        ok = top.code == code and top.formation == formation
        out.append(CaseResult(
            f"discovery-{field_name.replace(' ', '-').lower()}", ok,
            "ok" if ok else f"top pattern was {top.code} in {top.formation}, expected {code} in {formation}",
            {"top_code": top.code, "top_formation": top.formation,
             "hours": round(top.hours_total, 1), "wells_affected": top.wells_affected},
        ))

        brief = build_brief(store, f"{field_name}-NEXT", field_name, 3100.0, narrator=narrator)
        check = verify_brief(brief, store)
        out.append(CaseResult(
            f"brief-citations-{field_name.replace(' ', '-').lower()}", check["ok"],
            "ok" if check["ok"] else "; ".join(check["problems"][:3]),
            {"citations_checked": check["citations_checked"]},
        ))
        has_mitigation = any(r.mitigations for r in brief.risks)
        out.append(CaseResult(
            f"brief-mitigations-{field_name.replace(' ', '-').lower()}", has_mitigation,
            "ok" if has_mitigation else "no risk in the brief carried a written mitigation",
            {"risks": len(brief.risks)},
        ))
    return out


def run(store: Store, searcher: Searcher, narrator: Narrator | None = None) -> dict[str, Any]:
    narrator = narrator or OfflineNarrator()
    results: list[CaseResult] = [
        _run_retrieval_case(c, store, searcher, narrator) for c in RETRIEVAL_CASES
    ]
    results.extend(_check_arithmetic(store, searcher, narrator))
    results.append(_check_grounding(store, searcher, narrator))
    results.extend(_check_discovery(store, narrator))

    passed = sum(1 for r in results if r.passed)
    return {
        "narrator": narrator.name,
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "results": [
            {"case": r.case_id, "passed": r.passed, "detail": r.detail, "metrics": r.metrics}
            for r in results
        ],
    }
