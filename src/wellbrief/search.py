"""Hybrid retrieval: keyword, dense and structured filters fused together.

A drilling engineer's question is half prose and half database predicate.
"What went wrong in the 12 1/4 inch section on Orrindale" is a text query with
two hard filters buried in it. Running that as pure text search returns
Vessra South reports that happen to mention 12 1/4", which answers a different
question. So the query gets read for structure first, the candidate set is
narrowed on the parsed fields, and only then do BM25 and the dense index
compete inside that set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field

from .bm25 import BM25Index
from .config import DEFAULT_TOP_K, RRF_K
from .embed import Embedder
from .models import SearchHit
from .store import Store, VectorIndex
from .text import canonical_section, normalise, snippet, tokenize

# Phrases an engineer would type, mapped to the codes the reports use.
CODE_SYNONYMS: dict[str, list[str]] = {
    "STUCK_PIPE": ["stuck pipe", "stuck", "pack off", "packed off", "differential sticking"],
    "LOST_CIRCULATION": ["lost circulation", "losses", "total losses", "lost returns"],
    "WELLBORE_INSTABILITY": ["instability", "tight hole", "washout", "caving", "overpull"],
    "HOLE_CLEANING": ["hole cleaning", "cuttings bed", "back ream", "reaming"],
    "FISHING": ["fishing", "fish", "junk in hole"],
    "DOWNHOLE_TOOL_FAILURE": ["mwd", "tool failure", "downhole tool", "directional tool", "lwd"],
    "RIG_REPAIR": ["rig repair", "mud pump", "drawworks", "top drive", "equipment failure"],
    "BOP_TEST_FAILURE": ["bop", "preventer", "pressure test"],
    "CEMENT_ISSUE": ["cement", "cementing", "plug"],
    "WAIT_ON_MATERIALS": ["waiting on materials", "wom", "barite", "logistics"],
    "WAIT_ON_WEATHER": ["weather", "wow", "storm", "wind"],
    "THIRD_PARTY_STANDBY": ["standby", "third party", "wireline standby"],
    "HSE_STOP": ["hse", "stop work", "safety stand down"],
}


@dataclass
class QueryPlan:
    """What the system thinks the question is actually asking for."""

    text: str
    field_name: str | None = None
    wells: list[str] = dc_field(default_factory=list)
    doc_types: list[str] = dc_field(default_factory=list)
    hole_section: str | None = None
    formation: str | None = None
    codes: list[str] = dc_field(default_factory=list)
    depth_m: float | None = None

    def describe(self) -> str:
        bits = []
        if self.field_name:
            bits.append(f"field={self.field_name}")
        if self.wells:
            bits.append(f"wells={','.join(self.wells)}")
        if self.hole_section:
            bits.append(f"section={self.hole_section}")
        if self.formation:
            bits.append(f"formation={self.formation}")
        if self.codes:
            bits.append(f"codes={','.join(self.codes)}")
        if self.doc_types:
            bits.append(f"types={','.join(self.doc_types)}")
        if self.depth_m:
            bits.append(f"depth~{int(self.depth_m)}m")
        return " ".join(bits) or "no structural filter"


def plan_query(question: str, store: Store) -> QueryPlan:
    """Read filters out of a natural-language question."""
    raw = normalise(question)
    low = raw.lower()
    plan = QueryPlan(text=question)

    known_fields = {w.field_name for w in store.wells()}
    for fname in sorted(known_fields, key=len, reverse=True):
        if fname.lower() in low:
            plan.field_name = fname
            break

    for m in re.finditer(r"\b([A-Za-z]{2,5})-(\d{2,4})\b", raw):
        plan.wells.append(f"{m.group(1).upper()}-{m.group(2)}")

    m = re.search(r"(\d{1,2})\s*[-\s]?\s*(\d)/(\d)\s*(?:\"|in\b|inch\b)", raw, re.I)
    if not m:
        m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*(?:\"|inch\b)", raw)
    if m:
        plan.hole_section = canonical_section(m.group(0))

    formations = {f for w in store.wells() for f in w.formations}
    for fm in sorted(formations, key=len, reverse=True):
        if fm.lower() in low:
            plan.formation = fm
            break

    for code, phrases in CODE_SYNONYMS.items():
        if any(p in low for p in phrases):
            plan.codes.append(code)

    if re.search(r"\blesson|recommend|end of well|eowr\b", low):
        plan.doc_types.append("eowr")
    if re.search(r"\bincident|root cause|investigation\b", low):
        plan.doc_types.append("incident")

    m = re.search(r"\b(\d{1,3}(?:[,\s]\d{3})+|\d{3,5})\s*m\b", raw, re.I)
    if m:
        plan.depth_m = float(re.sub(r"[,\s]", "", m.group(1)))

    return plan


def _allowed_docs(plan: QueryPlan, store: Store) -> set[str] | None:
    """Narrow the candidate set using parsed fields, never using free text."""
    filters: dict = {}
    if plan.field_name:
        filters["field_name"] = plan.field_name
    if plan.wells:
        filters["well"] = plan.wells
    if plan.doc_types:
        filters["doc_type"] = plan.doc_types
    docs = store.documents(**filters) if filters else None
    allowed = None if docs is None else {d.doc_id for d in docs}

    # Section and formation live inside the DDR body, so they are applied
    # through the NPT ledger and the parsed metadata rather than SQL on text.
    if plan.hole_section or plan.formation or plan.codes:
        npt_filters: dict = {}
        if plan.field_name:
            npt_filters["field_name"] = plan.field_name
        if plan.wells:
            npt_filters["well"] = plan.wells
        if plan.hole_section:
            npt_filters["hole_section"] = plan.hole_section
        if plan.formation:
            npt_filters["formation"] = plan.formation
        if plan.codes:
            npt_filters["code"] = plan.codes
        matched = {e.doc_id for e in store.npt(**npt_filters)}
        if matched:
            # Keep the end-of-well and incident reports for the same wells:
            # that is where the mitigation language lives.
            wells = {e.well for e in store.npt(**npt_filters)}
            for d in store.documents(doc_type=["eowr", "incident"], well=sorted(wells)):
                matched.add(d.doc_id)
            allowed = matched if allowed is None else (allowed & matched) or matched
    return allowed


def rrf_fuse(rankings: dict[str, list[tuple[str, float]]],
             k: int = RRF_K) -> list[tuple[str, float, dict[str, float]]]:
    """Reciprocal rank fusion. Scale free, so BM25 and cosine can be mixed."""
    fused: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    for source, ranked in rankings.items():
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            contrib = 1.0 / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + contrib
            parts.setdefault(doc_id, {})[source] = round(score, 6)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_id, round(score, 6), parts.get(doc_id, {})) for doc_id, score in out]


class Searcher:
    def __init__(self, store: Store, bm25: BM25Index, vectors: VectorIndex, embedder: Embedder):
        self.store = store
        self.bm25 = bm25
        self.vectors = vectors
        self.embedder = embedder

    def search(self, question: str, top_k: int = DEFAULT_TOP_K,
               plan: QueryPlan | None = None) -> tuple[list[SearchHit], QueryPlan]:
        plan = plan or plan_query(question, self.store)
        allowed = _allowed_docs(plan, self.store)

        pool = max(top_k * 5, 40)
        lexical = self.bm25.search(question, top_k=pool, allowed=allowed)
        try:
            qvec = self.embedder.embed(question)
            dense = self.vectors.search(qvec, top_k=pool, allowed=allowed)
        except Exception:  # noqa: BLE001 - the dense side is optional
            # If the dense side raises, keyword results are still returned
            # on their own.
            dense = []

        fused = rrf_fuse({"bm25": lexical, "dense": dense})
        terms = tokenize(question)
        hits: list[SearchHit] = []
        for doc_id, score, parts in fused[:top_k]:
            doc = self.store.get_document(doc_id)
            if not doc:
                continue
            hits.append(SearchHit(
                doc_id=doc_id, score=score, document=doc,
                snippet=snippet(doc.text, terms), components=parts,
            ))
        return hits, plan


def build_searcher(store: Store, bm25: BM25Index, vectors: VectorIndex, embedder: Embedder) -> Searcher:
    return Searcher(store, bm25, vectors, embedder)
