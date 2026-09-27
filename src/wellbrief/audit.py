"""Audit log.

Every `ask` and `brief` invocation appends one JSON line to the workspace's
`audit.jsonl` (`Workspace.audit_path`): when it ran, what it was asked or
built, which corpus and narrator produced the answer, which documents it
cited, and whether citation verification passed. A line is written only
after the command actually produced a result -- never on an argument or
configuration error, since there is nothing to audit yet at that point.

The file is append-only and is never rewritten in place: each call opens it,
writes one line, and closes it, so earlier records survive even if a later
run is interrupted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import Answer, RiskBrief
    from .store import Store


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_ask(
    path: Path,
    *,
    question: str,
    fields: list[str],
    corpus_hash: str,
    narrator: str,
    cited_document_ids: list[str],
    verification_ok: bool,
    verification_reasons: int,
    abstained: bool,
) -> None:
    """Append one `ask` record. `verification_ok`/`verification_reasons` come from the
    answer's own citation warnings (an id the narrator cited that was not in the evidence
    pack): zero warnings is a passing verification."""
    _append(path, {
        "ts": _timestamp(),
        "command": "ask",
        "parameters": {"question": question, "fields": fields},
        "corpus_hash": corpus_hash,
        "narrator": narrator,
        "cited_document_ids": cited_document_ids,
        "verification": {"ok": verification_ok, "reasons": verification_reasons},
        "abstained": abstained,
    })


def log_brief(
    path: Path,
    *,
    parameters: dict[str, Any],
    corpus_hash: str,
    narrator: str,
    cited_document_ids: list[str],
    verification_ok: bool,
    verification_reasons: int,
) -> None:
    """Append one `brief` record, whether or not citation verification passed: a brief
    labelled `NOT VERIFIED` is still recorded, with its verification failure counted."""
    _append(path, {
        "ts": _timestamp(),
        "command": "brief",
        "parameters": parameters,
        "corpus_hash": corpus_hash,
        "narrator": narrator,
        "cited_document_ids": cited_document_ids,
        "verification": {"ok": verification_ok, "reasons": verification_reasons},
    })


def record_ask(path: Path, store: Store, question: str, answer: Answer) -> None:
    """Derive `log_ask`'s arguments from an `ask` result and append the record: the one call
    both `cli.cmd_ask` and the HTTP API's `POST /api/ask` make, so an `ask` run from either
    place is audited the same way."""
    fields = list(answer.query_plan.get("fields") or [])
    corpus_hash = store.corpus_hash(fields[0]) if len(fields) == 1 else store.corpus_hash()
    rejected_reasons = len(answer.narrator_rejected["reasons"]) if answer.narrator_rejected else 0
    log_ask(
        path,
        question=question,
        fields=fields,
        corpus_hash=corpus_hash,
        narrator=answer.narrator,
        cited_document_ids=list(dict.fromkeys(c.doc_id for c in answer.citations)),
        verification_ok=not answer.citation_warnings and not answer.narrator_rejected,
        verification_reasons=len(answer.citation_warnings) + rejected_reasons,
        abstained=answer.abstained,
    )


def record_brief(path: Path, brief: RiskBrief, verification: dict[str, Any],
                 parameters: dict[str, Any]) -> None:
    """Derive `log_brief`'s arguments from a `brief` result and its citation verification, and
    append the record: the one call both `cli.cmd_brief` and the HTTP API's brief endpoints
    make, so a brief built from either place is audited the same way."""
    rejected_reasons = len(brief.narrator_rejected["reasons"]) if brief.narrator_rejected else 0
    log_brief(
        path,
        parameters=parameters,
        corpus_hash=brief.provenance.get("corpus_hash", ""),
        narrator=brief.provenance.get("narrator", ""),
        cited_document_ids=list(dict.fromkeys(c.doc_id for r in brief.risks for c in r.citations)),
        verification_ok=verification["ok"] and not brief.narrator_rejected,
        verification_reasons=len(verification["problems"]) + rejected_reasons,
    )


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Every record in an audit log, in append order; an empty list when the file does
    not exist yet (no `ask` or `brief` has run in this workspace)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
