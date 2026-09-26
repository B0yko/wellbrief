"""Ground-truth sidecar of a generated corpus.

The generator knows what it planted and what it wrote. It records that in
`_truth.json` next to the documents: every document, every true NPT entry,
every well's planted-pattern membership, and a label for every candidate
sentence (end-of-well lessons and recommendations, incident corrective
actions). The labels come from the generator, from what it wrote; no
classifier is involved.

The leading underscore keeps ingestion from reading the file as a document.
The generator only writes it; the evaluation package is its only reader, so
gold answers stay independent of the product's own code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import SYNTHETIC_FOOTER
from .fields import (
    CARBONATE,
    G3_BHT_LIMIT_C,
    G3_SECTION,
    G3_TOOL,
    G4_RIG,
    MWD_TOOLS,
    MWD_VENDOR,
    OPERATOR,
    SALT,
    FieldSpec,
)
from .records import Corpus, Sentence, WellRecord

TRUTH_FILENAME = "_truth.json"
SCHEMA_VERSION = 1


def _planted_patterns(fields: tuple[FieldSpec, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pattern in ("G1", "G2", "G3", "G4"):
        for spec in fields:
            if pattern not in spec.patterns:
                continue
            if pattern == "G1":
                out.append({"id": "G1", "field": spec.name, "scope": "interval", "driver": "mud_weight",
                            "keys": [{"code": code, "section": spec.sections[1].size, "formation": SALT}
                                     for code in ("STUCK_PIPE", "FISHING", "WELLBORE_INSTABILITY")]})
            elif pattern == "G2":
                out.append({"id": "G2", "field": spec.name, "scope": "interval", "driver": "practice",
                            "keys": [{"code": "LOST_CIRCULATION", "section": spec.sections[-1].size,
                                      "formation": CARBONATE}]})
            elif pattern == "G3":
                out.append({"id": "G3", "field": spec.name, "scope": "equipment", "driver": "tool",
                            "keys": [{"code": "DOWNHOLE_TOOL_FAILURE", "mwd": G3_TOOL,
                                      "section": G3_SECTION}],
                            "bht_limit_c": G3_BHT_LIMIT_C})
            else:
                out.append({"id": "G4", "field": spec.name, "scope": "equipment", "driver": "rig",
                            "keys": [{"code": "RIG_REPAIR", "rig": G4_RIG}]})
    return out


def _sentence(doc_id: str, w: WellRecord, s: Sentence) -> dict[str, Any]:
    return {
        "doc_id": doc_id, "well": w.name, "field": w.spec.name, "kind": s.kind,
        "text": " ".join(s.text.split()), "label": s.label, "patterns": list(s.patterns), "code": s.code,
    }


def build_truth(corpus: Corpus, files: dict[str, str], formats: str) -> dict[str, Any]:
    """The sidecar as a JSON-ready dict. `files` maps doc id to the written file name."""
    fields = []
    for spec in corpus.fields:
        fields.append({
            "name": spec.name, "prefix": spec.prefix, "operator": OPERATOR,
            "rigs": list(corpus.rigs[spec.name]), "mwd_vendor": MWD_VENDOR, "mwd_tools": list(MWD_TOOLS),
            "patterns": sorted(spec.patterns),
            "surface_temp_c": spec.surface_temp_c, "geothermal_c_per_m": spec.geothermal_c_per_m,
            "sections": [{"size": s.size, "top_m": s.top_m, "base_m": s.base_m, "casing": s.casing,
                          "mud_system": s.mud_system} for s in spec.sections],
            "formations": [{"name": f.name, "top_m": f.top_m, "base_m": f.base_m, "lithology": f.lithology}
                           for f in spec.formations],
        })

    wells, documents, events, sentences = [], [], [], []
    for w in corpus.wells:
        spec = w.spec
        entry_ids = {id(e): f"{d.doc_id}#{n}" for d in w.days for n, e in enumerate(d.entries, start=1)}
        wells.append({
            "well": w.name, "field": spec.name, "rig": w.rig, "mwd": w.mwd,
            "mwd_by_section": dict(w.mwd_by_section),
            "spud": w.spud.isoformat(), "release": w.release.isoformat(), "td_m": w.td_m,
            "eowr_date": w.eowr_date.isoformat(), "days_on_well": len(w.days),
            "salt_mw_sg": w.salt_mw_sg,
            "salt_mw_ok": w.salt_mw_ok if "G1" in spec.patterns else None,
            "lcm_pretreat": w.lcm_pretreat if "G2" in spec.patterns else None,
            "patterns": dict(w.patterns),
            "cites": dict(w.citations),
        })
        for d in w.days:
            documents.append({
                "doc_id": d.doc_id, "type": "ddr", "field": spec.name, "well": w.name,
                "date": d.day.isoformat(), "file": files.get(d.doc_id),
                "sections": [d.section.size], "formations": list(d.formations),
                "report_no": d.report_no, "drilled": d.drilled,
                "depth_start_m": d.depth_start, "depth_end_m": d.depth_end,
                "formation_at_td": d.formation_end.name, "mud_weight_sg": d.mud_weight_sg, "bht_c": d.bht_c,
                "rig": w.rig, "mwd": d.mwd,
                "npt_hours": d.npt_tenths / 10, "productive_hours": (240 - d.npt_tenths) / 10,
            })
            for n, e in enumerate(d.entries, start=1):
                events.append({
                    "event_id": f"{d.doc_id}#{n}", "doc_id": d.doc_id, "well": w.name, "field": spec.name,
                    "date": d.day.isoformat(), "code": e.code, "hours": e.hours, "depth_m": e.depth_m,
                    "section": d.section.size, "formation": e.formation, "rig": w.rig, "mwd": d.mwd,
                    "pattern": e.pattern, "kind": e.kind, "variant": e.variant,
                    "occurrence": f"{w.name}#{e.event.number}", "part": e.part, "parts": len(e.event.parts),
                })
        documents.append({
            "doc_id": w.eowr_id, "type": "eowr", "field": spec.name, "well": w.name,
            "date": w.eowr_date.isoformat(), "file": files.get(w.eowr_id),
            "sections": list(dict.fromkeys(d.section.size for d in w.days)),
            "formations": list(dict.fromkeys(f for d in w.days for f in d.formations)),
            "rig": w.rig, "mwd": w.mwd,
        })
        for s in w.lessons + w.recommendations:
            sentences.append(_sentence(w.eowr_id, w, s))
        for inc in w.incidents:
            ev = inc.event
            start = w.report(inc.reports[0])
            ids = [entry_ids[id(p)] for p in ev.parts]
            documents.append({
                "doc_id": inc.doc_id, "type": "incident", "field": spec.name, "well": w.name,
                "date": start.day.isoformat(), "file": files.get(inc.doc_id),
                "sections": [start.section.size], "formations": [ev.formation],
                "code": ev.code, "hours": ev.hours, "depth_m": ev.depth_m,
                "section": start.section.size, "formation": ev.formation,
                "event_id": ids[0], "event_ids": ids, "occurrence": f"{w.name}#{ev.number}",
                "rig": w.rig, "mwd": start.mwd,
            })
            for s in inc.actions:
                sentences.append(_sentence(inc.doc_id, w, s))

    return {
        "schema": SCHEMA_VERSION,
        "seed": corpus.seed,
        "scale": corpus.scale,
        "formats": formats,
        "footer": SYNTHETIC_FOOTER,
        "fields": fields,
        "planted_patterns": _planted_patterns(corpus.fields),
        "wells": wells,
        "documents": documents,
        "npt_events": events,
        "sentences": sentences,
    }


def write_truth(out_dir: Path, truth: dict[str, Any]) -> Path:
    path = out_dir / TRUTH_FILENAME
    path.write_bytes((json.dumps(truth, indent=1, ensure_ascii=True) + "\n").encode("utf-8"))
    return path
