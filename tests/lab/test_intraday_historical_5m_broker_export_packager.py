from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.lab import intraday_historical_5m_broker_export_packager as packager


def config_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, run_validator: bool = False) -> packager.PackagerConfig:
    monkeypatch.setattr(packager, "REPO_ROOT", tmp_path)
    return packager.PackagerConfig(
        raw_export_dir=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"),
        manual_inbox=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox"),
        out_dir=Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager"),
        validator_out_dir=Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake"),
        run_manual_intake_validator=run_validator,
    )


def raw_dir(tmp_path: Path) -> Path:
    path = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_decision(tmp_path: Path) -> dict:
    path = tmp_path / ".local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager/broker_export_package_decision.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_raw_export_dir_missing_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = packager.package_exports(config_for(tmp_path, monkeypatch))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_BLOCKED_WAITING_FOR_RAW_EXPORT"
    assert read_decision(tmp_path)["package_generated"] is False


def test_valid_chinese_columns_csv_package_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "证券代码": "510300",
                "日期": "2026-03-02",
                "时间": "09:35",
                "开盘": 1.0,
                "最高": 1.1,
                "最低": 0.9,
                "收盘": 1.05,
                "成交量": 1000,
                "成交额": 1050,
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "510300_5m.csv", index=False, encoding="utf-8-sig")

    result = packager.package_exports(config_for(tmp_path, monkeypatch))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_READY_FOR_MANUAL_INTAKE_VALIDATOR"
    manual_csv = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/historical_5m_manual_export.csv"
    assert manual_csv.exists()
    packaged = pd.read_csv(manual_csv, dtype={"etf_code": str})
    assert packaged.loc[0, "etf_code"] == "510300"


def test_valid_english_columns_csv_package_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "159915",
                "trade_date": "2026-03-02",
                "datetime": "2026-03-02 09:35:00",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
                "amount": 1050,
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "159915_5m.csv", index=False)

    result = packager.package_exports(config_for(tmp_path, monkeypatch))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_READY_FOR_MANUAL_INTAKE_VALIDATOR"
    assert (tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/MANIFEST.json").exists()


def test_forbidden_account_order_trade_field_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "510300",
                "trade_date": "2026-03-02",
                "datetime": "2026-03-02 09:35:00",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
                "account": "blocked",
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "510300_5m.csv", index=False)

    result = packager.package_exports(config_for(tmp_path, monkeypatch))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_BLOCKED_FORBIDDEN_FIELDS"
    assert not (tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/historical_5m_manual_export.csv").exists()


def test_schema_unmappable_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(raw_dir(tmp_path) / "unknown.csv", index=False)

    result = packager.package_exports(config_for(tmp_path, monkeypatch))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_BLOCKED_SCHEMA_UNMAPPABLE"


def test_sha256sums_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "510500",
                "trade_date": "2026-03-02",
                "datetime": "2026-03-02 09:35:00",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "510500_5m.csv", index=False)

    packager.package_exports(config_for(tmp_path, monkeypatch))

    sha_path = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/SHA256SUMS.txt"
    lines = sha_path.read_text(encoding="utf-8").splitlines()
    assert any("historical_5m_manual_export.csv" in line for line in lines)
    assert any("source_note.md" in line for line in lines)
    assert any("MANIFEST.json" in line for line in lines)


def test_manifest_boundary_flags_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "512880",
                "trade_date": "2026-03-02",
                "datetime": "2026-03-02 09:35:00",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "512880_5m.csv", index=False)

    packager.package_exports(config_for(tmp_path, monkeypatch))

    manifest_path = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_allowed"] is False
    assert manifest["stable_effect_allowed"] is False
    assert manifest["contains_secret"] is False
    assert manifest["contains_order_intent"] is False
    assert manifest["contains_live_order"] is False
    assert manifest["contains_account"] is False
    assert manifest["contains_position"] is False
    assert manifest["contains_order"] is False
    assert manifest["contains_trade"] is False
    assert manifest["qmt_related"] is False
    assert manifest["qmt_mode"] == "not_qmt"


def test_output_path_outside_local_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packager, "REPO_ROOT", tmp_path)
    config = packager.PackagerConfig(
        raw_export_dir=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"),
        manual_inbox=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox"),
        out_dir=Path("docs/research/not_allowed"),
    )

    with pytest.raises(packager.BrokerExportPackagerError, match="path must be under .local_research_outputs"):
        packager.package_exports(config)


def test_validator_handoff_command_decision_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "588000",
                "trade_date": "2026-03-02",
                "datetime": "2026-03-02 09:35:00",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 1000,
            }
        ]
    ).to_csv(raw_dir(tmp_path) / "588000_5m.csv", index=False)

    def fake_validator(manual_inbox: Path, validator_out_dir: Path) -> dict:
        return {
            "command": ["fake-python", "tools/lab/intraday_historical_5m_manual_intake_validator.py", "--inbox", str(manual_inbox)],
            "returncode": 0,
            "manual_intake_validator_decision": "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION",
            "decision_path": str(validator_out_dir / "manual_intake_readiness_decision.json"),
        }

    monkeypatch.setattr(packager, "run_manual_validator", fake_validator)
    result = packager.package_exports(config_for(tmp_path, monkeypatch, run_validator=True))

    assert result["decision"] == "BROKER_EXPORT_PACKAGE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"
    decision = read_decision(tmp_path)
    assert decision["manual_intake_validator_decision"] == "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"
    assert "intraday_historical_5m_manual_intake_validator.py" in " ".join(decision["manual_intake_validator_command"])
