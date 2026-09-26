"""The check behind each case category, and the context the checks run in.

A check takes a case and the context of one seed (the generated corpus, its
truth database and a product workspace built from it) and returns an
`Outcome`. Gold answers come only from the truth database or from the corpus
files; the product is reached only through `adapter`.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import adapter
from .cases import Case
from .metrics import TOLERANCES, match_counts, mrr_at_k, precision_at_k, squash, verbatim, within
from .truth import Truth

NO_MATCH = "No records in this workspace match that question."
RETRIEVAL_K = 8
MIN_PRECISION_AT_K = 0.5
# The categories `--no-risk-filters` applies to. Under that flag their cases are
# reported without a pass or fail (the targets are set for the filtered brief);
# every other brief of the run keeps its filters.
FILTER_FLAG_CATEGORIES = frozenset({"brief-precision"})


@dataclass
class Outcome:
    passed: bool
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


class Context:
    """Everything the checks of one seed share. The workspace is built on first use."""

    def __init__(self, seed: int, corpus_dir: Path, work_dir: Path, truth: Truth,
                 risk_filters: bool = True) -> None:
        self.seed = seed
        self.corpus_dir = corpus_dir
        self.work_dir = work_dir
        self.truth = truth
        self.risk_filters = risk_filters
        self.cases: dict[str, Case] = {}
        self._workspace: adapter.Workspace | None = None
        self._workspace_error: Exception | None = None
        self._asks: dict[tuple[Any, ...], adapter.AskResult] = {}
        self._briefs: dict[tuple[Any, ...], adapter.Brief] = {}

    @property
    def workspace(self) -> adapter.Workspace:
        if self._workspace_error is not None:
            raise self._workspace_error
        if self._workspace is None:
            try:
                self._workspace = adapter.Workspace(self.corpus_dir, self.work_dir / "workspace")
            except Exception as exc:
                self._workspace_error = exc
                raise
        return self._workspace

    def ask(self, question: str, top_k: int = RETRIEVAL_K, spread_rate: float | None = None) -> adapter.AskResult:
        key = (question, top_k, spread_rate)
        if key not in self._asks:
            self._asks[key] = self.workspace.ask(question, top_k=top_k, spread_rate=spread_rate)
        return self._asks[key]

    def filters_for(self, case: Case) -> bool:
        """Whether the case's brief keeps the risk filters (`--no-risk-filters` reaches brief precision only).

        A case whose brief is built without them is reported, not scored.
        """
        return self.risk_filters or case.category not in FILTER_FLAG_CATEGORIES

    def brief(self, case: Case) -> adapter.Brief:
        filters = self.filters_for(case)
        key = (case["field"], case["well"], float(case["td_m"]), filters)
        if key not in self._briefs:
            self._briefs[key] = self.workspace.brief(case["field"], case["well"], float(case["td_m"]), filters)
        return self._briefs[key]

    def source_text(self, doc_id: str) -> str | None:
        """The text of a generated document, read from its file (None if the corpus has no such document)."""
        doc = self.truth.doc(doc_id)
        if doc is None or not doc["file"]:
            return None
        return (self.corpus_dir / str(doc["file"])).read_text(encoding="utf-8")

    def doc_field(self, doc_id: str) -> str | None:
        doc = self.truth.doc(doc_id)
        return None if doc is None else str(doc["field"])

    def close(self) -> None:
        if self._workspace is not None:
            self._workspace.close()


def _fail(detail: str, **metrics: Any) -> Outcome:
    return Outcome(False, detail, metrics)


def _outcome(problems: Sequence[str], ok_detail: str = "ok", **metrics: Any) -> Outcome:
    if problems:
        return Outcome(False, "; ".join(problems[:3]) + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""),
                       metrics)
    return Outcome(True, ok_detail, metrics)


# ---------------------------------------------------------------------------
# Answers: retrieval, arithmetic, grounding, abstention
# ---------------------------------------------------------------------------

def check_retrieval(case: Case, ctx: Context) -> Outcome:
    """Prototype rule: right field, a cited report from a well that had the code, the doc type, no warnings."""
    answer = ctx.ask(case["question"], case.get("top_k", RETRIEVAL_K))
    docs = [c.doc_id for c in answer.citations]
    if not docs:
        return _fail("no documents retrieved")
    rows = {d: ctx.truth.doc(d) for d in docs}
    known = {d: r for d, r in rows.items() if r is not None}
    if len(known) < len(rows):
        return _fail(f"cited documents that are not in the corpus: {sorted(set(rows) - set(known))[:3]}",
                     retrieved=len(docs))
    wrong = [d for d in docs if known[d]["field"] != case["field"]]
    if wrong:
        return _fail(f"retrieved documents from another field: {wrong[:3]}", retrieved=len(docs))
    if case.get("gold_sql"):
        wells = set(ctx.truth.column(case["gold_sql"]))
        if not wells:
            return _fail("the truth has no well with the asked-about code")
        matched = sum(1 for d in docs if known[d]["well"] in wells)
        if matched < case.get("min_hits", 1):
            return _fail("no retrieved document belongs to a well that had the code",
                         retrieved=len(docs), matched=matched)
    if case.get("doc_type"):
        types = sorted({str(known[d]["type"]) for d in docs})
        if case["doc_type"] not in types:
            return _fail(f"expected a {case['doc_type']} among the cited documents, got {types}")
    if answer.warnings:
        return _fail(f"citation warnings: {answer.warnings[:2]}")
    return Outcome(True, "ok", {"retrieved": len(docs)})


def check_retrieval_precision(case: Case, ctx: Context) -> Outcome:
    relevant = set(ctx.truth.column(case["relevant_sql"]))
    if not relevant:
        return _fail("relevant_sql returned no documents on this corpus")
    answer = ctx.ask(case["question"])
    top = answer.ranking[:RETRIEVAL_K]
    p = precision_at_k(top, relevant, RETRIEVAL_K)
    mrr = mrr_at_k(top, relevant, RETRIEVAL_K)
    shown = list(dict.fromkeys(top + [c.doc_id for c in answer.citations]))
    foreign = [d for d in shown if ctx.doc_field(d) != case["field"]]
    metrics = {"p_at_8": round(p, 4), "mrr_at_8": round(mrr, 4), "relevant": len(relevant),
               "hits": sum(1 for d in top if d in relevant), "ranking": top}
    problems = []
    if p < MIN_PRECISION_AT_K:
        problems.append(f"P@8 {p:.2f} < {MIN_PRECISION_AT_K}")
    if foreign:
        problems.append(f"documents from another field or not in the corpus: {foreign[:3]}")
    detail = (f"P@8 {p:.2f} ({metrics['hits']} of {min(RETRIEVAL_K, len(relevant))}), MRR@8 {mrr:.2f}, "
              f"{len(relevant)} relevant")
    if problems:
        return Outcome(False, f"{detail}; " + "; ".join(problems), metrics)
    return Outcome(True, detail, metrics)


def check_arithmetic(case: Case, ctx: Context) -> Outcome:
    figure = case["figure"]
    spread_rate = case.get("spread_rate")
    params = {"spread_rate": spread_rate} if spread_rate is not None else {}
    gold = ctx.truth.scalar(case["gold_sql"], params)
    if gold is None:
        return _fail("gold_sql returned no value on this corpus")
    answer = ctx.ask(case["question"], spread_rate=spread_rate)
    got = answer.figures.get(figure)
    metrics = {"expected": gold, "reported": got, "figure": figure}
    if got is None:
        return Outcome(False, f"the answer reports no {figure} (expected {gold:g})", metrics)
    if within(got, float(gold), TOLERANCES[figure]):
        return Outcome(True, f"{figure} {got:g} (gold {gold:g})", metrics)
    return Outcome(False, f"expected {figure} {gold:g}, the answer reported {got:g}", metrics)


def check_grounding(case: Case, ctx: Context) -> Outcome:
    problems: list[str] = []
    checked = 0
    for ref in case["from_cases"]:
        answer = ctx.ask(ctx.cases[ref]["question"], case.get("top_k", RETRIEVAL_K))
        for cite in answer.citations:
            checked += 1
            text = ctx.source_text(cite.doc_id)
            if text is None:
                problems.append(f"{ref}: {cite.doc_id} does not exist")
            elif not verbatim(cite.quote, text):
                problems.append(f"{ref}: quote not verbatim in {cite.doc_id}")
    return _outcome(problems, f"{checked} citations verbatim", citations_checked=checked, not_verbatim=len(problems))


def check_abstention(case: Case, ctx: Context) -> Outcome:
    answer = ctx.ask(case["question"])
    problems = []
    if answer.figures:
        problems.append(f"the answer carries figures {sorted(answer.figures)}")
    if answer.citations:
        problems.append(f"the answer cites {len(answer.citations)} documents")
    if not any(line.strip() == NO_MATCH for line in answer.text.splitlines()):
        first = answer.text.strip().splitlines()[0][:80] if answer.text.strip() else ""
        problems.append(f"the no-match sentence is missing (answer starts: {first!r})")
    if answer.abstained is False:
        problems.append("the product reports that it did not abstain")
    return _outcome(problems, "abstained", figures=len(answer.figures), citations=len(answer.citations))


# ---------------------------------------------------------------------------
# Patterns and briefs
# ---------------------------------------------------------------------------

def pattern_keys(truth: Truth, field_name: str, pattern: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT id, code, section, formation, rig, mwd, driver FROM truth_patterns WHERE field = ?"
    params: tuple[Any, ...] = (field_name,)
    if pattern is not None:
        sql += " AND id = ?"
        params += (pattern,)
    return [dict(r) for r in truth.rows(sql, params)]


def names(product_value: str | None, truth_value: str | None) -> bool:
    """Does the product's name of a rig or tool name the truth's one?

    The reports give a tool with its vendor ("Parvane Downhole PJ-3") where
    the truth file keeps the model ("PJ-3"), so the truth name must appear in
    the product's as a whole token: "PJ-3" names neither "PJ-35" nor "PJ-5".
    """
    if not product_value or not truth_value:
        return False
    return re.search(rf"(?<![\w-]){re.escape(truth_value)}(?![\w-])", product_value) is not None


def risk_matches(risk: adapter.Risk, key: dict[str, Any]) -> bool:
    """Does a listed risk match a planted-pattern key of the truth file?

    Interval keys (G1, G2) need an interval risk with the same code, hole
    section and formation. The rig key (G4) needs a risk keyed on that rig.
    The tool key (G3) accepts two forms: a risk keyed on the PJ-3 tool, or an
    interval risk with the same code in the 12 1/4" section, whatever its
    formation (the tool fails where the hole is hot, which spans formations).
    """
    if risk.code != key["code"]:
        return False
    if key["rig"]:
        return names(risk.rig, key["rig"])
    if key["mwd"]:
        return names(risk.mwd, key["mwd"]) or bool(risk.scope == "interval" and risk.section == key["section"])
    return bool(risk.scope == "interval" and risk.section == key["section"] and risk.formation == key["formation"])


def describe(risk: adapter.Risk) -> str:
    where = risk.rig or risk.mwd or f"{risk.section} {risk.formation}"
    return f"{risk.code} ({risk.scope}: {where})"


def _matching(brief: adapter.Brief, keys: list[dict[str, Any]]) -> list[tuple[int, adapter.Risk]]:
    return [(i, r) for i, r in enumerate(brief.risks, start=1) if any(risk_matches(r, k) for k in keys)]


def _planted(case: Case, ctx: Context) -> tuple[list[dict[str, Any]], adapter.Brief]:
    keys = pattern_keys(ctx.truth, case["field"], case["pattern"])
    if not keys:
        raise ValueError(f"the truth file plants no {case['pattern']} in {case['field']}")
    return keys, ctx.brief(case)


def check_discovery(case: Case, ctx: Context) -> Outcome:
    patterns = ctx.workspace.patterns(case["field"])
    if not patterns:
        return _fail("no patterns found")
    top = patterns[0]
    metrics = {"top_code": top.code, "top_formation": top.formation, "hours": round(top.hours, 1)}
    if top.code == case["code"] and top.formation == case["formation"]:
        return Outcome(True, f"top pattern {top.code} in {top.formation}", metrics)
    return Outcome(False, f"top pattern was {top.code} in {top.formation}, expected {case['code']} in "
                          f"{case['formation']}", metrics)


def check_brief_citations(case: Case, ctx: Context) -> Outcome:
    brief = ctx.brief(case)
    problems: list[str] = []
    checked = 0
    for risk in brief.risks:
        for cite in risk.citations:
            checked += 1
            text = ctx.source_text(cite.doc_id)
            if text is None:
                problems.append(f"{risk.code}: unknown document {cite.doc_id}")
            elif not verbatim(cite.quote, text):
                problems.append(f"{risk.code}: quote not verbatim in {cite.doc_id}")
    if not brief.verified:
        problems.append(f"the product marks the brief as not verified: {brief.problems[:2]}")
    return _outcome(problems, f"{checked} citations verbatim", citations_checked=checked)


def check_brief_mitigations(case: Case, ctx: Context) -> Outcome:
    brief = ctx.brief(case)
    count = sum(len(r.mitigations) for r in brief.risks)
    if count:
        return Outcome(True, f"{count} mitigations across {len(brief.risks)} risks", {"risks": len(brief.risks)})
    return _fail("no risk in the brief carried a written mitigation", risks=len(brief.risks))


def check_brief_recall(case: Case, ctx: Context) -> Outcome:
    keys, brief = _planted(case, ctx)
    found = _matching(brief, keys)
    if found:
        rank, risk = found[0]
        return Outcome(True, f"listed as {describe(risk)}, rank {rank} of {len(brief.risks)}", {"rank": rank})
    listed = [describe(r) for r in brief.risks]
    return _fail(f"not among the {len(listed)} listed risks: {listed}", listed=listed)


def check_brief_driver(case: Case, ctx: Context) -> Outcome:
    """The driver kind the truth gives, and for a rig or tool driver the rig or tool of the pattern's key."""
    keys, brief = _planted(case, ctx)
    driver = keys[0]["driver"]
    found = _matching(brief, keys)
    if not found:
        return _fail(f"{case['pattern']} is not listed, so it has no driver (expected {driver})")
    risk = found[0][1]
    key = next(k for k in keys if risk_matches(risk, k))
    blamed = key["rig"] or key["mwd"]
    expected = f"{driver} ({blamed})" if blamed else driver
    reported = f"{risk.driver_kind} ({risk.driver_category})" if risk.driver_category else risk.driver_kind
    metrics = {"expected": expected, "reported": reported, "driver_text": risk.driver}
    if risk.driver_kind == driver and (blamed is None or names(risk.driver_category, blamed)):
        return Outcome(True, f"{describe(risk)} driver: {expected}", metrics)
    return Outcome(False, f"{describe(risk)} driver is {reported or 'not stated'}, expected {expected}", metrics)


def _containing(truth: Truth, doc_id: str, text: str) -> list[dict[str, Any]]:
    """Truth candidate sentences of `doc_id` that contain the quoted mitigation."""
    quoted = squash(text)
    if not quoted:
        return []
    rows = truth.rows("SELECT kind, text, label, patterns, code FROM truth_sentences WHERE doc_id = ?", (doc_id,))
    return [dict(r) for r in rows if quoted in r["text"]]


def clean_wells(truth: Truth, field_name: str, risk: adapter.Risk, planted: set[str]) -> set[str]:
    """Wells of the field that were exposed to the risk and did not have it, from the true events.

    The miner's definition: for an interval risk, a well
    with a daily report in the same section and formation and no event of the
    risk's code there; for an equipment risk, a well of the field on another
    rig (or with another MWD tool in the 12 1/4" section) that did not have the
    event. For an equipment risk that matches planted patterns (`planted`),
    "the event" is an event of those patterns, so unrelated events of the same
    code on the other rig (drawworks repairs next to the mud pump pattern) do
    not disqualify a well; for any other equipment risk it is any event of the
    risk's code.
    """
    if risk.rig or risk.mwd:
        column, value = ("rig", risk.rig) if risk.rig else ("mwd", risk.mwd)
        others = [r["well"] for r in truth.rows(f"SELECT well, {column} AS unit FROM truth_wells WHERE field = ?",
                                                 (field_name,)) if not names(value, r["unit"])]
        if planted:
            marks = ", ".join("?" * len(planted))
            had = truth.column(f"SELECT well FROM truth_events WHERE field = ? AND pattern IN ({marks})",
                               (field_name, *sorted(planted)))
        else:
            had = truth.column("SELECT well FROM truth_events WHERE field = ? AND code = ?", (field_name, risk.code))
        return set(others) - set(had)
    exposed = truth.column(
        "SELECT d.well FROM truth_docs d, json_each(d.sections) s, json_each(d.formations) f "
        "WHERE d.type = 'ddr' AND d.field = ? AND s.value = ? AND f.value = ?",
        (field_name, risk.section, risk.formation))
    had = truth.column("SELECT well FROM truth_events WHERE field = ? AND code = ? AND section = ? AND formation = ?",
                       (field_name, risk.code, risk.section, risk.formation))
    return set(exposed) - set(had)


def _relates(sentence: dict[str, Any], risk: adapter.Risk, planted: set[str]) -> bool:
    """A sentence relates to a risk when the truth gives it the risk's code or a pattern the risk matches."""
    return bool(sentence["code"] == risk.code or planted & set(json.loads(sentence["patterns"])))


def mitigation_problem(truth: Truth, field_name: str, risk: adapter.Risk, m: adapter.Mitigation) -> str | None:
    doc = truth.doc(m.doc_id)
    if doc is None:
        return "the source is not a corpus document"
    found = _containing(truth, m.doc_id, m.text)
    if not found:
        return "not a lesson, recommendation or corrective action of its source"
    labels = sorted({s["label"] for s in found})
    if labels != ["practice"]:
        return f"labelled {'/'.join(labels)} in the truth file"
    planted = {k["id"] for k in pattern_keys(truth, field_name) if risk_matches(risk, k)}
    if not any(_relates(s, risk, planted) for s in found):
        return "the truth relates the sentence to another code or pattern"
    if doc["type"] == "incident":
        return None if doc["code"] == risk.code else f"incident report of another code ({doc['code']})"
    if doc["type"] == "eowr":
        clean = clean_wells(truth, field_name, risk, planted)
        return None if doc["well"] in clean else f"{doc['well']} did not avoid it"
    return f"the source is a {doc['type']}"


def check_mitigation_precision(case: Case, ctx: Context) -> Outcome:
    brief = ctx.brief(case)
    quoted = [(r, m) for r in brief.risks for m in r.mitigations]
    if not quoted:
        return _fail("the brief quotes no mitigations", mitigations=0)
    problems = []
    for risk, m in quoted:
        why = mitigation_problem(ctx.truth, case["field"], risk, m)
        if why:
            problems.append(f"{risk.code} [{m.doc_id}] {why}: {m.text[:50]!r}")
    precise = len(quoted) - len(problems)
    metrics = {"mitigations": len(quoted), "precise": precise, "precision": round(precise / len(quoted), 4)}
    return _outcome(problems, f"{len(quoted)} of {len(quoted)} mitigations are related practice from a valid source",
                    **metrics)


def check_mitigation_recall(case: Case, ctx: Context) -> Outcome:
    keys, brief = _planted(case, ctx)
    found = _matching(brief, keys)
    if not found:
        return _fail(f"{case['pattern']} is not listed")
    quoted = [m for _, r in found for m in r.mitigations]
    good = [m for m in quoted
            if any(s["label"] == "practice" and case["pattern"] in json.loads(s["patterns"])
                   for s in _containing(ctx.truth, m.doc_id, m.text))]
    metrics = {"mitigations": len(quoted), "practice_for_pattern": len(good)}
    if good:
        return Outcome(True, f"{len(good)} of {len(quoted)} mitigations are {case['pattern']} practice", metrics)
    return Outcome(False, f"none of the {len(quoted)} mitigations is a {case['pattern']} practice sentence", metrics)


def check_brief_precision(case: Case, ctx: Context) -> Outcome:
    """Share of listed risks that match a planted key; a pattern split over several risks counts each of them."""
    brief = ctx.brief(case)
    filters = ctx.filters_for(case)
    if not brief.risks:
        return _fail(f"the brief lists no risks (filters {'on' if filters else 'off'})", risk_filters=filters)
    keys = pattern_keys(ctx.truth, case["field"])
    planted = [r for r in brief.risks if any(risk_matches(r, k) for k in keys)]
    covered = sorted({k["id"] for r in planted for k in keys if risk_matches(r, k)})
    precision = len(planted) / len(brief.risks)
    others = [describe(r) for r in brief.risks if r not in planted]
    metrics = {"listed": len(brief.risks), "planted": len(planted), "precision": round(precision, 4),
               "patterns_covered": covered, "risk_filters": filters, "not_planted": others}
    detail = (f"{len(planted)} of {len(brief.risks)} listed risks are planted ({precision:.2f}; patterns "
              f"{', '.join(covered) or 'none'}; filters {'on' if filters else 'off'}")
    if not filters:
        return Outcome(True, detail + "; reported, not scored)", metrics)
    return Outcome(precision >= case["min_precision"], detail + f"; target {case['min_precision']:.2f})", metrics)


# ---------------------------------------------------------------------------
# Ledger, renderings and classifier
# ---------------------------------------------------------------------------

def check_parser_fidelity(case: Case, ctx: Context) -> Outcome:
    """The parsed ledger against the true events: rows match on document, code and hours."""
    gold = [dict(r) for r in ctx.truth.rows(case["gold_sql"])]
    if not gold:
        return _fail("gold_sql returned no events on this corpus")
    parsed = ctx.workspace.ledger(case["field"])
    pool: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for g in gold:
        pool.setdefault((g["doc_id"], g["code"], round(g["hours"], 1)), []).append(g)
    matched = 0
    agree = {"depth": 0, "section": 0, "formation": 0}
    for r in parsed:
        candidates = pool.get((r.doc_id, r.code, round(r.hours, 1)))
        if not candidates:
            continue
        g = candidates.pop()
        matched += 1
        agree["depth"] += int(r.depth_m is not None and abs(r.depth_m - g["depth_m"]) < 0.5)
        agree["section"] += int(r.section == g["section"])
        agree["formation"] += int(r.formation == g["formation"])
    precision = matched / len(parsed) if parsed else 0.0
    recall = matched / len(gold)
    total_parsed = round(sum(r.hours for r in parsed), 1)
    total_gold = round(sum(g["hours"] for g in gold), 1)
    metrics = {"precision": round(precision, 4), "recall": round(recall, 4), "parsed": len(parsed),
               "truth": len(gold), "total_hours_parsed": total_parsed, "total_hours_truth": total_gold,
               **{f"{k}_agree": v for k, v in agree.items()}}
    detail = (f"precision {precision:.3f}, recall {recall:.3f}, {total_parsed} h vs {total_gold} h; depth/section/"
              f"formation agree on {agree['depth']}/{agree['section']}/{agree['formation']} of {matched}")
    passed = precision == 1.0 and recall == 1.0 and within(total_parsed, total_gold, TOLERANCES["total_hours"])
    return Outcome(passed, detail, metrics)


def _parity_key(fmt: str) -> Callable[[adapter.LedgerRow], tuple[Any, ...]]:
    """A CSV row becomes its own document, so its rows are compared without the document id."""
    if fmt == "csv":
        return lambda r: (r.well, r.date, r.code, r.hours, r.depth_m, r.section, r.formation)
    return lambda r: (r.doc_id, r.code, r.hours, r.depth_m, r.section, r.formation)


def check_format_parity(case: Case, ctx: Context) -> Outcome:
    """Same ledger rows as the txt rendering, field by field over the whole workspace (rows filed under no
    field or a wrong field count), and every question answered with quotes that verify: verbatim in the txt
    rendering of the cited document, or for the csv rendering a whole line of the ledger file."""
    fmt = case["format"]
    root = ctx.work_dir / f"render-{fmt}"
    shutil.rmtree(root, ignore_errors=True)
    folder = adapter.render(root, ctx.seed, fmt)
    rendered = adapter.Workspace(folder, root / "workspace")
    try:
        problems: list[str] = []
        key = _parity_key(fmt)
        base, other = ctx.workspace.ledger(None), rendered.ledger(None)
        for name in sorted({r.field or "" for r in base} | {r.field or "" for r in other}):
            matched, n_other, n_base = match_counts((key(r) for r in other if (r.field or "") == name),
                                                    (key(r) for r in base if (r.field or "") == name))
            if not matched == n_other == n_base:
                problems.append(f"{name or 'no field'}: {matched} of {n_base} txt ledger rows reproduced, "
                                f"{n_other} rows")
        csv_lines = set()
        if fmt == "csv":
            csv_text = (folder / "npt-ledger.csv").read_text(encoding="utf-8")
            csv_lines = {squash(line) for line in csv_text.splitlines() if line.strip()}
        checked = 0
        for question in case["questions"]:
            citations = rendered.ask(question).citations
            if not citations:
                problems.append(f"no citations for {question!r}")
            for cite in citations:
                checked += 1
                if fmt == "csv":
                    if squash(cite.quote) not in csv_lines:
                        problems.append(f"quote of {cite.doc_id} is not a whole ledger row")
                elif not verbatim(cite.quote, ctx.source_text(cite.doc_id) or ""):
                    problems.append(f"quote not verbatim in {cite.doc_id}")
        return _outcome(problems, f"ledger rows identical, {checked} quotes verified", citations_checked=checked,
                        rows=len(other))
    finally:
        rendered.close()


def check_classifier_accuracy(case: Case, ctx: Context) -> Outcome:
    """The product's label for every candidate sentence against the truth label.

    Only the sentence texts go to the product. Every candidate sentence counts
    once per document it appears in (a recommendation repeated in 20 EOWRs
    counts 20 times); accuracy over distinct texts is reported next to it.
    Practice precision (the share of sentences called practice that are
    practice) is what keeps failure narratives out of the mitigations.
    """
    rows = [dict(r) for r in ctx.truth.rows(case["gold_sql"])]
    if not rows:
        return _fail("gold_sql returned no sentences on this corpus")
    predicted = ctx.workspace.classify([r["text"] for r in rows])
    if len(predicted) != len(rows):
        return _fail(f"the classifier returned {len(predicted)} labels for {len(rows)} sentences")
    pairs = list(zip((r["label"] for r in rows), predicted, strict=True))
    correct = sum(1 for truth_label, p in pairs if truth_label == p)
    accuracy = correct / len(rows)
    distinct = {(r["text"], r["label"]): p for r, p in zip(rows, predicted, strict=True)}
    distinct_accuracy = sum(1 for (_, label), p in distinct.items() if label == p) / len(distinct)
    said_practice = [label for label, p in pairs if p == "practice"]
    practice_precision = said_practice.count("practice") / len(said_practice) if said_practice else 0.0
    failures = [p for label, p in pairs if label == "failure"]
    metrics = {"sentences": len(rows), "distinct_sentences": len(distinct), "correct": correct,
               "accuracy": round(accuracy, 4), "distinct_accuracy": round(distinct_accuracy, 4),
               "practice_precision": round(practice_precision, 4),
               "failure_recall": round(failures.count("failure") / len(failures), 4) if failures else None,
               "failure_called_practice": failures.count("practice")}
    passed = accuracy >= case["min_accuracy"]
    detail = f"accuracy {accuracy:.3f} ({correct}/{len(rows)}), target {case['min_accuracy']:.2f}"
    if case.get("min_practice_precision") is not None:
        passed = passed and practice_precision >= case["min_practice_precision"]
        detail += f"; practice precision {practice_precision:.3f}, target {case['min_practice_precision']:.2f}"
    return Outcome(passed, detail, metrics)


CHECKS: dict[str, Callable[[Case, Context], Outcome]] = {
    "retrieval": check_retrieval,
    "arithmetic": check_arithmetic,
    "grounding": check_grounding,
    "discovery": check_discovery,
    "brief-citations": check_brief_citations,
    "brief-mitigations": check_brief_mitigations,
    "retrieval-precision": check_retrieval_precision,
    "abstention": check_abstention,
    "brief-recall": check_brief_recall,
    "brief-driver": check_brief_driver,
    "mitigation-precision": check_mitigation_precision,
    "mitigation-recall": check_mitigation_recall,
    "parser-fidelity": check_parser_fidelity,
    "format-parity": check_format_parity,
    "brief-precision": check_brief_precision,
    "classifier-accuracy": check_classifier_accuracy,
}
