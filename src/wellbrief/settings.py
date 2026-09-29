"""Configuration precedence: the one place every module reads settings from.

`wellbrief.toml` (a workspace's own, or one named by `ingest --config`, or one
sitting in a folder being ingested) holds the tables below; every value in a
table is optional, and any key this module does not recognise -- a whole
table, a table's own key, or (for `parse.*`, `detect.headings`, `csv.columns`)
one of that table's *logical* keys -- is a `SettingsError` naming the file and
the key, not a silently ignored typo.

Tables:

- `[parse.ddr.labels]`, `[parse.eowr.sections]`, `[parse.incident.sections]`:
  the label and section-heading text each parser looks for (`parse.py`).
- `[detect.headings]`: the content heading (`detect.py`) that names each
  document type, keyed `ddr` / `eowr` / `incident`.
- `[taxonomy.aliases]`: a site's own NPT code spelling mapped to the built-in
  taxonomy (`config.NPT_CODES`); `[taxonomy.keywords]`: the mitigation
  miner's relevance words, per code.
- `[csv]` (`date_format`) and `[csv.columns]`: the NPT ledger CSV's own
  header names (`readers.csvledger`).
- `[retrieval]` (`top_k`, `rrf_k`, `k1`, `b`) and `[risk]` (`min_lift`,
  `min_wells`, `min_support`, `max_risks`, `equipment_min_ratio`,
  `equipment_min_rate`, `equipment_min_wells`).

Precedence, highest first: CLI flags; the environment variables
`WELLBRIEF_HOME`, `WELLBRIEF_WORKSPACE`, `WELLBRIEF_SPREAD_RATE_USD_PER_DAY`;
`ingest --config PATH` or (when `--config` is not given) the ingested
folder's own `wellbrief.toml`; `<workspace>/wellbrief.toml`; the built-in
defaults in `config.py`. Only `home`, `workspace_name` and `spread_rate` have
an environment variable here (the narrator and LLM ones, `WELLBRIEF_LLM_*`, are read by the
CLI and `egress.py`, not by this module); the parse, detection, taxonomy, CSV, retrieval and risk tables are
`wellbrief.toml`-only, so for them the precedence collapses to CLI (`top_k`,
`max_risks`, where a command exposes one) > ingest config/folder toml (parse,
detection, taxonomy, CSV only -- these matter only while ingesting) >
workspace toml > defaults.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NoReturn

from . import config as defaults

__all__ = [
    "CsvSettings",
    "DetectSettings",
    "ParseSettings",
    "RetrievalSettings",
    "RiskSettings",
    "Settings",
    "SettingsError",
    "TaxonomySettings",
    "default_settings",
    "env_home",
    "env_spread_rate",
    "env_workspace_name",
    "load_toml_document",
    "resolve_home",
    "resolve_settings",
    "resolve_spread_rate",
    "resolve_workspace_name",
]


class SettingsError(ValueError):
    """A `wellbrief.toml` (or `ingest --config` file) names an unknown table or key, or gives
    a value of the wrong shape. Always names the file and the offending key."""


ENV_HOME = "WELLBRIEF_HOME"
ENV_WORKSPACE = "WELLBRIEF_WORKSPACE"
ENV_SPREAD_RATE = "WELLBRIEF_SPREAD_RATE_USD_PER_DAY"

DEFAULT_HOME = Path.home() / ".wellbrief"
DEFAULT_WORKSPACE_NAME = "default"

# The NPT ledger CSV's logical columns (`readers.csvledger.LOGICAL_COLUMNS`), duplicated here
# (rather than imported) so this module never has to import a reader module just to validate a
# TOML table; the two lists are kept in step by `test_settings.py`.
_CSV_LOGICAL_COLUMNS = ("well", "date", "code", "hours", "field", "depth", "section", "formation",
                        "rig", "description")


# ---------------------------------------------------------------------------
# Resolved settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseSettings:
    """`[parse.ddr.labels]`, `[parse.eowr.sections]`, `[parse.incident.sections]`."""

    ddr_labels: dict[str, str] = field(default_factory=lambda: dict(defaults.DEFAULT_DDR_LABELS))
    eowr_sections: dict[str, str] = field(default_factory=lambda: dict(defaults.DEFAULT_EOWR_SECTIONS))
    incident_sections: dict[str, str] = field(
        default_factory=lambda: dict(defaults.DEFAULT_INCIDENT_SECTIONS))


@dataclass(frozen=True)
class DetectSettings:
    """`[detect.headings]`: document type -> the content heading that names it."""

    headings: dict[str, str] = field(default_factory=lambda: dict(defaults.DEFAULT_DETECT_HEADINGS))


@dataclass(frozen=True)
class TaxonomySettings:
    """`[taxonomy.aliases]` (a site's own NPT code -> the built-in taxonomy) and
    `[taxonomy.keywords]` (the mitigation miner's relevance words, per code)."""

    aliases: dict[str, str] = field(default_factory=lambda: dict(defaults.DEFAULT_TAXONOMY_ALIASES))
    keywords: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in defaults.TAXONOMY_KEYWORDS.items()})


@dataclass(frozen=True)
class CsvSettings:
    """`[csv]` `date_format` and `[csv.columns]`."""

    columns: dict[str, str] = field(default_factory=lambda: dict(defaults.DEFAULT_CSV_COLUMNS))
    date_format: str | None = defaults.DEFAULT_CSV_DATE_FORMAT


@dataclass(frozen=True)
class RetrievalSettings:
    """`[retrieval]`."""

    top_k: int = defaults.DEFAULT_TOP_K
    rrf_k: int = defaults.RRF_K
    k1: float = defaults.BM25_K1
    b: float = defaults.BM25_B


@dataclass(frozen=True)
class RiskSettings:
    """`[risk]`."""

    min_lift: float = defaults.RISK_MIN_LIFT
    min_wells: int = defaults.RISK_MIN_WELLS
    min_support: float = defaults.RISK_MIN_SUPPORT
    max_risks: int = defaults.RISK_MAX_RISKS
    equipment_min_ratio: float = defaults.EQUIPMENT_MIN_RATIO
    equipment_min_rate: float = defaults.EQUIPMENT_MIN_RATE
    equipment_min_wells: int = defaults.EQUIPMENT_MIN_WELLS


@dataclass(frozen=True)
class Settings:
    """Everything a run needs, fully resolved: no module below this one reads
    `os.environ` or a TOML file of its own."""

    home: Path = DEFAULT_HOME
    workspace_name: str = DEFAULT_WORKSPACE_NAME
    spread_rate_usd_per_day: float = defaults.DEFAULT_SPREAD_RATE_USD_PER_DAY
    parse: ParseSettings = field(default_factory=ParseSettings)
    detect: DetectSettings = field(default_factory=DetectSettings)
    taxonomy: TaxonomySettings = field(default_factory=TaxonomySettings)
    csv: CsvSettings = field(default_factory=CsvSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    sources: tuple[str, ...] = ()   # the wellbrief.toml files folded in, lowest precedence first

    def ingest_parse_config(self) -> dict[str, Any]:
        """The subset of settings that decide how a file is parsed, in a plain, JSON-serialisable
        shape: what `ingest.ingest_folder` records on every `FileRecord` it writes, so the parse
        configuration used at ingest time is recorded per file and re-parsing stays reproducible.
        Retrieval and risk settings are not parsing decisions, so they are not
        part of it."""
        return {
            "parse": {
                "ddr_labels": dict(self.parse.ddr_labels),
                "eowr_sections": dict(self.parse.eowr_sections),
                "incident_sections": dict(self.parse.incident_sections),
            },
            "detect_headings": dict(self.detect.headings),
            "taxonomy_aliases": dict(self.taxonomy.aliases),
            "csv": {"columns": dict(self.csv.columns), "date_format": self.csv.date_format},
        }


def default_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Environment variables (the only os.environ reads in the whole codebase)
# ---------------------------------------------------------------------------


def env_home(env: dict[str, str] | None = None) -> Path | None:
    raw = (env if env is not None else os.environ).get(ENV_HOME)
    return Path(raw) if raw else None


def env_workspace_name(env: dict[str, str] | None = None) -> str | None:
    return (env if env is not None else os.environ).get(ENV_WORKSPACE) or None


def env_spread_rate(env: dict[str, str] | None = None) -> float | None:
    raw = (env if env is not None else os.environ).get(ENV_SPREAD_RATE)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise SettingsError(f"{ENV_SPREAD_RATE}={raw!r} is not a number") from None


def resolve_home(explicit: str | os.PathLike[str] | None = None,
                 env: dict[str, str] | None = None) -> Path:
    """`$WELLBRIEF_HOME`: `explicit` (a CLI flag), else the environment variable, else
    `~/.wellbrief`."""
    if explicit:
        return Path(explicit)
    return env_home(env) or DEFAULT_HOME


def resolve_workspace_name(explicit: str | None = None, env: dict[str, str] | None = None) -> str:
    """The workspace name: `--workspace` (`explicit`), else `WELLBRIEF_WORKSPACE`, else `"default"`."""
    return explicit or env_workspace_name(env) or DEFAULT_WORKSPACE_NAME


def resolve_spread_rate(explicit: float | None = None, env: dict[str, str] | None = None) -> float:
    """The spread rate in USD/day: `--spread-rate` (`explicit`), else
    `WELLBRIEF_SPREAD_RATE_USD_PER_DAY`, else the built-in default."""
    if explicit is not None:
        return explicit
    env_value = env_spread_rate(env)
    return env_value if env_value is not None else defaults.DEFAULT_SPREAD_RATE_USD_PER_DAY


# ---------------------------------------------------------------------------
# TOML: validation and per-table overrides
# ---------------------------------------------------------------------------


@dataclass
class _Overrides:
    """What one `wellbrief.toml` document actually set; every field is `None` (or empty)
    where the document said nothing, so folding several of these together (lowest precedence
    first) never clobbers a key a higher-precedence document left unmentioned."""

    ddr_labels: dict[str, str] = field(default_factory=dict)
    eowr_sections: dict[str, str] = field(default_factory=dict)
    incident_sections: dict[str, str] = field(default_factory=dict)
    detect_headings: dict[str, str] = field(default_factory=dict)
    taxonomy_aliases: dict[str, str] = field(default_factory=dict)
    taxonomy_keywords: dict[str, list[str]] = field(default_factory=dict)
    csv_columns: dict[str, str] = field(default_factory=dict)
    csv_date_format: str | None = None
    # Mixed int (`top_k`/`min_wells`/...) and float (`k1`/`min_lift`/...) values: `Any`, not
    # `float`, so folding them onto `RetrievalSettings`/`RiskSettings` with `**overrides` does
    # not make every field there look like it could receive a float.
    retrieval: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)


def _fail(source: str, path: str, message: str) -> NoReturn:
    raise SettingsError(f"{source}: {path}: {message}")


def _expect_table(value: Any, source: str, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(source, path, f"must be a table, not {type(value).__name__}")
    return value


def _expect_keys(table: dict[str, Any], allowed: tuple[str, ...], source: str, path: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        _fail(source, path, f"unknown key(s) {', '.join(unknown)}; expected one of {', '.join(allowed)}")


def _expect_str(value: Any, source: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(source, path, "must be a non-empty string")
    return value


def _expect_str_map(table: dict[str, Any], allowed: tuple[str, ...], source: str,
                    path: str) -> dict[str, str]:
    """A table whose keys are drawn from `allowed` and whose values are non-empty strings."""
    _expect_keys(table, allowed, source, path)
    return {k: _expect_str(v, source, f"{path}.{k}") for k, v in table.items()}


def _expect_number(value: Any, source: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(source, path, f"must be a number, not {type(value).__name__}")
    return float(value)


def _expect_int(value: Any, source: str, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(source, path, f"must be an integer, not {type(value).__name__}")
    return value


def _validate_parse(
    table: dict[str, Any], source: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    _expect_keys(table, ("ddr", "eowr", "incident"), source, "parse")
    ddr_labels: dict[str, str] = {}
    if "ddr" in table:
        ddr = _expect_table(table["ddr"], source, "parse.ddr")
        _expect_keys(ddr, ("labels",), source, "parse.ddr")
        if "labels" in ddr:
            labels = _expect_table(ddr["labels"], source, "parse.ddr.labels")
            ddr_labels = _expect_str_map(labels, tuple(defaults.DEFAULT_DDR_LABELS), source,
                                         "parse.ddr.labels")
    eowr_sections: dict[str, str] = {}
    if "eowr" in table:
        eowr = _expect_table(table["eowr"], source, "parse.eowr")
        _expect_keys(eowr, ("sections",), source, "parse.eowr")
        if "sections" in eowr:
            sections = _expect_table(eowr["sections"], source, "parse.eowr.sections")
            eowr_sections = _expect_str_map(sections, tuple(defaults.DEFAULT_EOWR_SECTIONS), source,
                                            "parse.eowr.sections")
    incident_sections: dict[str, str] = {}
    if "incident" in table:
        incident = _expect_table(table["incident"], source, "parse.incident")
        _expect_keys(incident, ("sections",), source, "parse.incident")
        if "sections" in incident:
            sections = _expect_table(incident["sections"], source, "parse.incident.sections")
            incident_sections = _expect_str_map(sections, tuple(defaults.DEFAULT_INCIDENT_SECTIONS),
                                                source, "parse.incident.sections")
    return ddr_labels, eowr_sections, incident_sections


def _validate_detect(table: dict[str, Any], source: str) -> dict[str, str]:
    _expect_keys(table, ("headings",), source, "detect")
    if "headings" not in table:
        return {}
    headings = _expect_table(table["headings"], source, "detect.headings")
    return _expect_str_map(headings, tuple(defaults.DEFAULT_DETECT_HEADINGS), source, "detect.headings")


def _validate_taxonomy(table: dict[str, Any], source: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    _expect_keys(table, ("aliases", "keywords"), source, "taxonomy")
    known_codes = tuple(defaults.NPT_CODES)
    aliases: dict[str, str] = {}
    if "aliases" in table:
        raw = _expect_table(table["aliases"], source, "taxonomy.aliases")
        for site_code, taxonomy_code in raw.items():
            value = _expect_str(taxonomy_code, source, f"taxonomy.aliases.{site_code}")
            if value not in known_codes:
                _fail(source, f"taxonomy.aliases.{site_code}",
                      f"{value!r} is not a taxonomy code; expected one of {', '.join(known_codes)}")
            aliases[site_code] = value
    keywords: dict[str, list[str]] = {}
    if "keywords" in table:
        raw = _expect_table(table["keywords"], source, "taxonomy.keywords")
        _expect_keys(raw, known_codes, source, "taxonomy.keywords")
        for code, words in raw.items():
            path = f"taxonomy.keywords.{code}"
            if not isinstance(words, list) or not words:
                _fail(source, path, "must be a non-empty list of strings")
            keywords[code] = [_expect_str(w, source, f"{path}[{i}]") for i, w in enumerate(words)]
    return aliases, keywords


def _validate_csv(table: dict[str, Any], source: str) -> tuple[dict[str, str], str | None]:
    _expect_keys(table, ("date_format", "columns"), source, "csv")
    date_format = None
    if "date_format" in table:
        date_format = _expect_str(table["date_format"], source, "csv.date_format")
    columns: dict[str, str] = {}
    if "columns" in table:
        raw = _expect_table(table["columns"], source, "csv.columns")
        columns = _expect_str_map(raw, _CSV_LOGICAL_COLUMNS, source, "csv.columns")
    return columns, date_format


def _validate_retrieval(table: dict[str, Any], source: str) -> dict[str, Any]:
    _expect_keys(table, ("top_k", "rrf_k", "k1", "b"), source, "retrieval")
    out: dict[str, float] = {}
    if "top_k" in table:
        out["top_k"] = _expect_int(table["top_k"], source, "retrieval.top_k")
    if "rrf_k" in table:
        out["rrf_k"] = _expect_int(table["rrf_k"], source, "retrieval.rrf_k")
    if "k1" in table:
        out["k1"] = _expect_number(table["k1"], source, "retrieval.k1")
    if "b" in table:
        out["b"] = _expect_number(table["b"], source, "retrieval.b")
    return out


_RISK_INT_KEYS = ("min_wells", "max_risks", "equipment_min_wells")
_RISK_FLOAT_KEYS = ("min_lift", "min_support", "equipment_min_ratio", "equipment_min_rate")


def _validate_risk(table: dict[str, Any], source: str) -> dict[str, Any]:
    _expect_keys(table, _RISK_INT_KEYS + _RISK_FLOAT_KEYS, source, "risk")
    out: dict[str, float] = {}
    for key in _RISK_INT_KEYS:
        if key in table:
            out[key] = _expect_int(table[key], source, f"risk.{key}")
    for key in _RISK_FLOAT_KEYS:
        if key in table:
            out[key] = _expect_number(table[key], source, f"risk.{key}")
    return out


_TOP_LEVEL_TABLES = ("parse", "detect", "taxonomy", "csv", "retrieval", "risk")


def load_toml_document(data: dict[str, Any], *, source: str) -> _Overrides:
    """Validate one already-parsed TOML document (a `dict`) and return what it overrides.
    `source` names the document in every error message (a file path, or a label for a test)."""
    _expect_keys(data, _TOP_LEVEL_TABLES, source, "<document>")
    out = _Overrides()
    if "parse" in data:
        out.ddr_labels, out.eowr_sections, out.incident_sections = _validate_parse(
            _expect_table(data["parse"], source, "parse"), source)
    if "detect" in data:
        out.detect_headings = _validate_detect(_expect_table(data["detect"], source, "detect"), source)
    if "taxonomy" in data:
        out.taxonomy_aliases, out.taxonomy_keywords = _validate_taxonomy(
            _expect_table(data["taxonomy"], source, "taxonomy"), source)
    if "csv" in data:
        out.csv_columns, out.csv_date_format = _validate_csv(
            _expect_table(data["csv"], source, "csv"), source)
    if "retrieval" in data:
        out.retrieval = _validate_retrieval(_expect_table(data["retrieval"], source, "retrieval"), source)
    if "risk" in data:
        out.risk = _validate_risk(_expect_table(data["risk"], source, "risk"), source)
    return out


def load_toml_file(path: Path) -> _Overrides:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SettingsError(f"{path}: cannot read: {exc}") from exc
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{path}: invalid TOML: {exc}") from exc
    return load_toml_document(data, source=str(path))


def _fold(layers: list[_Overrides]) -> _Overrides:
    """Fold `layers` (lowest precedence first) into one: dict-shaped tables merge key by key
    (a later layer's key wins, wholesale -- a code's whole keyword list, not just an addition
    to it), and a scalar (`csv_date_format`) is replaced outright by the last layer that set it."""
    out = _Overrides()
    for layer in layers:
        out.ddr_labels.update(layer.ddr_labels)
        out.eowr_sections.update(layer.eowr_sections)
        out.incident_sections.update(layer.incident_sections)
        out.detect_headings.update(layer.detect_headings)
        out.taxonomy_aliases.update(layer.taxonomy_aliases)
        out.taxonomy_keywords.update(layer.taxonomy_keywords)
        out.csv_columns.update(layer.csv_columns)
        out.retrieval.update(layer.retrieval)
        out.risk.update(layer.risk)
        if layer.csv_date_format is not None:
            out.csv_date_format = layer.csv_date_format
    return out


def _apply(base: Settings, overrides: _Overrides, *, sources: tuple[str, ...]) -> Settings:
    parse = ParseSettings(
        ddr_labels={**base.parse.ddr_labels, **overrides.ddr_labels},
        eowr_sections={**base.parse.eowr_sections, **overrides.eowr_sections},
        incident_sections={**base.parse.incident_sections, **overrides.incident_sections},
    )
    detect = DetectSettings(headings={**base.detect.headings, **overrides.detect_headings})
    taxonomy = TaxonomySettings(
        aliases={**base.taxonomy.aliases, **overrides.taxonomy_aliases},
        keywords={**base.taxonomy.keywords, **overrides.taxonomy_keywords},
    )
    csv = CsvSettings(
        columns={**base.csv.columns, **overrides.csv_columns},
        date_format=overrides.csv_date_format if overrides.csv_date_format is not None
        else base.csv.date_format,
    )
    retrieval = replace(base.retrieval, **overrides.retrieval)
    risk = replace(base.risk, **overrides.risk)
    return replace(base, parse=parse, detect=detect, taxonomy=taxonomy, csv=csv,
                  retrieval=retrieval, risk=risk, sources=sources)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_settings(
    *,
    home: str | os.PathLike[str] | None = None,
    workspace_name: str | None = None,
    spread_rate: float | None = None,
    top_k: int | None = None,
    max_risks: int | None = None,
    workspace_root: str | os.PathLike[str] | None = None,
    ingest_config_path: str | os.PathLike[str] | None = None,
    ingest_folder: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> Settings:
    """Resolve one `Settings`, the precedence the module docstring describes.

    `home`/`workspace_name`/`spread_rate`/`top_k`/`max_risks` are a CLI's own explicit flags
    (`None` when not given, so the layer below decides); `workspace_root` is the workspace
    directory whose own `wellbrief.toml` (if any) is the lowest TOML layer; `ingest_config_path`
    (`ingest --config`) or, failing that, `ingest_folder`'s own `wellbrief.toml` is the layer
    above it -- both only matter to `ingest`, so a command with neither (`ask`, `brief`, ...)
    simply does not have this layer. `env` overrides `os.environ` for testing.
    """
    layers: list[_Overrides] = []
    sources: list[str] = []

    if workspace_root is not None:
        ws_toml = Path(workspace_root) / "wellbrief.toml"
        if ws_toml.is_file():
            layers.append(load_toml_file(ws_toml))
            sources.append(str(ws_toml))

    if ingest_config_path is not None:
        layers.append(load_toml_file(Path(ingest_config_path)))
        sources.append(str(ingest_config_path))
    elif ingest_folder is not None:
        folder_toml = Path(ingest_folder) / "wellbrief.toml"
        if folder_toml.is_file():
            layers.append(load_toml_file(folder_toml))
            sources.append(str(folder_toml))

    settings = _apply(Settings(), _fold(layers), sources=tuple(sources))
    return replace(
        settings,
        home=resolve_home(home, env),
        workspace_name=resolve_workspace_name(workspace_name, env),
        spread_rate_usd_per_day=resolve_spread_rate(spread_rate, env),
        retrieval=replace(settings.retrieval, **({"top_k": top_k} if top_k is not None else {})),
        risk=replace(settings.risk, **({"max_risks": max_risks} if max_risks is not None else {})),
    )
