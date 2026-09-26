"""`settings.py`: TOML validation, and the precedence CLI flags > environment > `ingest --config` /
the ingested folder's own `wellbrief.toml` > `<workspace>/wellbrief.toml` > built-in defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from wellbrief import config
from wellbrief.readers.csvledger import LOGICAL_COLUMNS
from wellbrief.settings import (
    _CSV_LOGICAL_COLUMNS,
    ENV_HOME,
    ENV_SPREAD_RATE,
    ENV_WORKSPACE,
    SettingsError,
    default_settings,
    load_toml_document,
    resolve_home,
    resolve_settings,
    resolve_spread_rate,
    resolve_workspace_name,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_settings_matches_the_built_in_config_constants() -> None:
    st = default_settings()
    assert st.spread_rate_usd_per_day == config.DEFAULT_SPREAD_RATE_USD_PER_DAY
    assert st.parse.ddr_labels == config.DEFAULT_DDR_LABELS
    assert st.parse.eowr_sections == config.DEFAULT_EOWR_SECTIONS
    assert st.parse.incident_sections == config.DEFAULT_INCIDENT_SECTIONS
    assert st.detect.headings == config.DEFAULT_DETECT_HEADINGS
    assert st.taxonomy.aliases == config.DEFAULT_TAXONOMY_ALIASES
    assert st.taxonomy.keywords == config.TAXONOMY_KEYWORDS
    assert st.csv.columns == config.DEFAULT_CSV_COLUMNS
    assert st.csv.date_format is None
    assert (st.retrieval.top_k, st.retrieval.rrf_k, st.retrieval.k1, st.retrieval.b) == (
        config.DEFAULT_TOP_K, config.RRF_K, config.BM25_K1, config.BM25_B)
    assert st.risk.max_risks == config.RISK_MAX_RISKS
    assert st.risk.min_lift == config.RISK_MIN_LIFT


def test_csv_logical_columns_agree_with_the_reader() -> None:
    """`settings._CSV_LOGICAL_COLUMNS` is a standalone copy so this module never imports a
    reader just to validate `[csv.columns]`; this test is what keeps the two in step."""
    assert set(_CSV_LOGICAL_COLUMNS) == set(LOGICAL_COLUMNS)


# ---------------------------------------------------------------------------
# TOML validation: unknown tables and keys are errors, values are shape-checked
# ---------------------------------------------------------------------------


def test_unknown_top_level_table_is_an_error() -> None:
    with pytest.raises(SettingsError, match=r"unknown key.*narrator"):
        load_toml_document({"narrator": {}}, source="<test>")


def test_unknown_key_in_a_known_table_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"retrieval": {"tpo_k": 5}}, source="<test>")


def test_unknown_key_in_parse_ddr_labels_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"parse": {"ddr": {"labels": {"not_a_real_field": "X"}}}}, source="<test>")


def test_unknown_subtable_under_parse_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"parse": {"csv": {}}}, source="<test>")


def test_unknown_key_in_detect_headings_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"detect": {"headings": {"other": "X"}}}, source="<test>")


def test_taxonomy_alias_to_an_unknown_code_is_an_error() -> None:
    with pytest.raises(SettingsError, match="not a taxonomy code"):
        load_toml_document({"taxonomy": {"aliases": {"SP": "NOT_A_CODE"}}}, source="<test>")


def test_taxonomy_keywords_for_an_unknown_code_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"taxonomy": {"keywords": {"NOT_A_CODE": ["x"]}}}, source="<test>")


def test_taxonomy_keywords_value_must_be_a_non_empty_list() -> None:
    with pytest.raises(SettingsError, match="non-empty list"):
        load_toml_document({"taxonomy": {"keywords": {"STUCK_PIPE": []}}}, source="<test>")


def test_unknown_csv_column_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"csv": {"columns": {"operator": "Operator"}}}, source="<test>")


def test_csv_date_format_must_be_a_string() -> None:
    with pytest.raises(SettingsError, match="non-empty string"):
        load_toml_document({"csv": {"date_format": 1}}, source="<test>")


def test_retrieval_value_must_be_a_number() -> None:
    with pytest.raises(SettingsError, match="must be a number"):
        load_toml_document({"retrieval": {"k1": "fast"}}, source="<test>")


def test_retrieval_top_k_must_be_an_integer_not_a_float() -> None:
    with pytest.raises(SettingsError, match="must be an integer"):
        load_toml_document({"retrieval": {"top_k": 8.5}}, source="<test>")


def test_risk_unknown_key_is_an_error() -> None:
    with pytest.raises(SettingsError, match="unknown key"):
        load_toml_document({"risk": {"min_lft": 2.0}}, source="<test>")


def test_a_bool_is_not_accepted_where_a_number_or_integer_is_expected() -> None:
    with pytest.raises(SettingsError, match="must be a number"):
        load_toml_document({"retrieval": {"k1": True}}, source="<test>")
    with pytest.raises(SettingsError, match="must be an integer"):
        load_toml_document({"risk": {"min_wells": True}}, source="<test>")


def test_valid_document_folds_every_table() -> None:
    doc = {
        "parse": {
            "ddr": {"labels": {"weight": "MW"}},
            "eowr": {"sections": {"lessons": "LESSONS"}},
            "incident": {"sections": {"root_cause": "CAUSE"}},
        },
        "detect": {"headings": {"ddr": "DAILY REPORT"}},
        "taxonomy": {"aliases": {"SP": "STUCK_PIPE"}, "keywords": {"STUCK_PIPE": ["seized"]}},
        "csv": {"date_format": "%d.%m.%Y", "columns": {"well": "Well Name"}},
        "retrieval": {"top_k": 5, "rrf_k": 40, "k1": 1.2, "b": 0.8},
        "risk": {"min_lift": 3.0, "min_wells": 4, "min_support": 0.5, "max_risks": 6,
                 "equipment_min_ratio": 1.5, "equipment_min_rate": 0.5, "equipment_min_wells": 2},
    }
    ov = load_toml_document(doc, source="<test>")
    assert ov.ddr_labels == {"weight": "MW"}
    assert ov.eowr_sections == {"lessons": "LESSONS"}
    assert ov.incident_sections == {"root_cause": "CAUSE"}
    assert ov.detect_headings == {"ddr": "DAILY REPORT"}
    assert ov.taxonomy_aliases == {"SP": "STUCK_PIPE"}
    assert ov.taxonomy_keywords == {"STUCK_PIPE": ["seized"]}
    assert ov.csv_columns == {"well": "Well Name"}
    assert ov.csv_date_format == "%d.%m.%Y"
    assert ov.retrieval == {"top_k": 5, "rrf_k": 40, "k1": 1.2, "b": 0.8}
    assert ov.risk["min_wells"] == 4 and ov.risk["max_risks"] == 6


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_resolve_home_precedence() -> None:
    assert resolve_home(env={}) == Path.home() / ".wellbrief"
    assert resolve_home(env={ENV_HOME: "/tmp/x"}) == Path("/tmp/x")
    assert resolve_home("/explicit", env={ENV_HOME: "/tmp/x"}) == Path("/explicit")


def test_resolve_workspace_name_precedence() -> None:
    assert resolve_workspace_name(env={}) == "default"
    assert resolve_workspace_name(env={ENV_WORKSPACE: "from-env"}) == "from-env"
    assert resolve_workspace_name("from-flag", env={ENV_WORKSPACE: "from-env"}) == "from-flag"


def test_resolve_spread_rate_precedence() -> None:
    assert resolve_spread_rate(env={}) == config.DEFAULT_SPREAD_RATE_USD_PER_DAY
    assert resolve_spread_rate(env={ENV_SPREAD_RATE: "60000"}) == 60000.0
    assert resolve_spread_rate(75000.0, env={ENV_SPREAD_RATE: "60000"}) == 75000.0


def test_bad_spread_rate_env_value_is_a_settings_error() -> None:
    with pytest.raises(SettingsError):
        resolve_spread_rate(env={ENV_SPREAD_RATE: "not-a-number"})


# ---------------------------------------------------------------------------
# Full resolution: workspace toml < ingest config/folder toml < CLI
# ---------------------------------------------------------------------------


def _write_toml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_workspace_toml_overrides_defaults(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_toml(ws / "wellbrief.toml", "[retrieval]\ntop_k = 12\n\n[risk]\nmax_risks = 3\n")
    st = resolve_settings(workspace_root=ws)
    assert st.retrieval.top_k == 12
    assert st.risk.max_risks == 3
    # untouched keys keep the built-in default
    assert st.retrieval.rrf_k == config.RRF_K
    assert st.risk.min_lift == config.RISK_MIN_LIFT
    assert st.sources == (str(ws / "wellbrief.toml"),)


def test_cli_flag_overrides_workspace_toml(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_toml(ws / "wellbrief.toml", "[retrieval]\ntop_k = 12\n")
    st = resolve_settings(workspace_root=ws, top_k=20)
    assert st.retrieval.top_k == 20


def test_ingest_folders_own_toml_overrides_workspace_toml(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_toml(ws / "wellbrief.toml", "[parse.ddr.labels]\nweight = \"From workspace\"\n")
    folder = tmp_path / "site"
    folder.mkdir()
    _write_toml(folder / "wellbrief.toml", "[parse.ddr.labels]\nweight = \"From folder\"\n"
                                          "hole_section = \"Hole size\"\n")
    st = resolve_settings(workspace_root=ws, ingest_folder=folder)
    assert st.parse.ddr_labels["weight"] == "From folder"
    assert st.parse.ddr_labels["hole_section"] == "Hole size"
    # a key the folder's own toml did not mention still falls back through the workspace's
    assert st.parse.ddr_labels["formation_at_td"] == config.DEFAULT_DDR_LABELS["formation_at_td"]


def test_ingest_config_flag_overrides_the_folders_own_toml(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    folder = tmp_path / "site"
    folder.mkdir()
    _write_toml(folder / "wellbrief.toml", "[parse.ddr.labels]\nweight = \"From folder\"\n")
    config_path = tmp_path / "explicit.toml"
    _write_toml(config_path, "[parse.ddr.labels]\nweight = \"From --config\"\n")
    st = resolve_settings(workspace_root=ws, ingest_config_path=config_path, ingest_folder=folder)
    assert st.parse.ddr_labels["weight"] == "From --config"


def test_no_wellbrief_toml_anywhere_is_fine(tmp_path: Path) -> None:
    st = resolve_settings(workspace_root=tmp_path / "nonexistent-workspace",
                          ingest_folder=tmp_path / "nonexistent-folder")
    assert st.parse.ddr_labels == config.DEFAULT_DDR_LABELS
    assert st.sources == ()


def test_a_bad_toml_file_names_itself_in_the_error(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    bad = ws / "wellbrief.toml"
    _write_toml(bad, "[risk]\nmin_lft = 2.0\n")
    with pytest.raises(SettingsError, match=str(bad)):
        resolve_settings(workspace_root=ws)


def test_ingest_parse_config_excludes_retrieval_and_risk(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_toml(ws / "wellbrief.toml", "[retrieval]\ntop_k = 3\n[taxonomy.aliases]\nSP = \"STUCK_PIPE\"\n")
    st = resolve_settings(workspace_root=ws)
    blob = st.ingest_parse_config()
    assert "retrieval" not in blob and "risk" not in blob
    assert blob["taxonomy_aliases"] == {"SP": "STUCK_PIPE"}
