"""A small, complete truth sidecar for evaluation tests, built record by record."""

from __future__ import annotations

from typing import Any

STATUSES = {"G1": "out_of_scope", "G2": "out_of_scope", "G3": "out_of_scope", "G4": "out_of_scope"}


def sidecar(*, fields: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": 1, "seed": 1, "scale": 1, "formats": "txt", "footer": "footer",
        "fields": fields or [field("Orrindale", "ORD", ["Orrin-1"], ["PJ-3"], ["G1"])],
        "planted_patterns": [], "wells": [], "documents": [], "npt_events": [], "sentences": [],
    }


def field(name: str, prefix: str, rigs: list[str], tools: list[str], patterns: list[str]) -> dict[str, Any]:
    return {"name": name, "prefix": prefix, "operator": "Operator", "rigs": rigs, "mwd_tools": tools,
            "patterns": patterns}


def well(name: str, field_name: str, rig: str, mwd: str = "PJ-3", **statuses: str) -> dict[str, Any]:
    return {"well": name, "field": field_name, "rig": rig, "mwd": mwd, "mwd_by_section": {},
            "spud": "2022-01-01", "release": "2022-02-01", "td_m": 3000, "eowr_date": "2022-02-10",
            "days_on_well": 31, "salt_mw_sg": None, "salt_mw_ok": None, "lcm_pretreat": None, "cites": {},
            "patterns": {**STATUSES, **statuses}}


def ddr(doc_id: str, w: dict[str, Any], section: str, formation: str) -> dict[str, Any]:
    return {"doc_id": doc_id, "type": "ddr", "field": w["field"], "well": w["well"], "date": "2022-01-02",
            "file": f"{doc_id}.txt", "sections": [section], "formations": [formation], "rig": w["rig"],
            "mwd": w["mwd"], "report_no": 1, "drilled": True, "depth_start_m": 0.0, "depth_end_m": 100.0,
            "formation_at_td": formation, "mud_weight_sg": 1.2, "bht_c": 50.0, "npt_hours": 0.0,
            "productive_hours": 24.0}


def eowr(doc_id: str, w: dict[str, Any]) -> dict[str, Any]:
    return {"doc_id": doc_id, "type": "eowr", "field": w["field"], "well": w["well"], "date": "2022-02-10",
            "file": f"{doc_id}.txt", "sections": [], "formations": [], "rig": w["rig"], "mwd": w["mwd"]}


def event(event_id: str, doc: dict[str, Any], code: str, hours: float, pattern: str | None = None,
          variant: str | None = None) -> dict[str, Any]:
    return {"event_id": event_id, "doc_id": doc["doc_id"], "well": doc["well"], "field": doc["field"],
            "date": doc["date"], "code": code, "hours": hours, "depth_m": 50.0, "section": doc["sections"][0],
            "formation": doc["formations"][0], "rig": doc["rig"], "mwd": doc["mwd"], "pattern": pattern,
            "kind": "npt", "variant": variant, "occurrence": event_id, "part": 1, "parts": 1}


def sentence(doc: dict[str, Any], text: str, label: str, code: str | None, patterns: list[str],
             kind: str = "lesson") -> dict[str, Any]:
    return {"doc_id": doc["doc_id"], "well": doc["well"], "field": doc["field"], "kind": kind, "text": text,
            "label": label, "patterns": patterns, "code": code}
