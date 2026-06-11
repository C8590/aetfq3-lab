from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tools.lab import intraday_broker_client_local_data_rescue as rescue


def config_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, roots: list[Path] | None = None) -> rescue.RescueConfig:
    monkeypatch.setattr(rescue, "REPO_ROOT", tmp_path)
    raw = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"
    raw.mkdir(parents=True, exist_ok=True)
    return rescue.RescueConfig(
        local_roots=roots or [raw],
        export_roots=[raw],
        raw_export_dir=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"),
        out_dir=Path(".local_research_outputs/aetfq3_lab/intraday_broker_client_local_data_rescue"),
        rescued_dir=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports_rescued"),
        manual_inbox=Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox"),
        packager_out_dir=Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager"),
        validator_out_dir=Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake"),
        run_packager=False,
    )


def test_empty_export_csv_classified_as_header_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path, monkeypatch)
    raw = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"
    (raw / "510300_5m.csv").write_text("日期,时间,开盘,最高,最低,收盘,成交量\n", encoding="utf-8-sig")

    frame = rescue.diagnose_empty_exports(config.export_roots)

    row = frame[frame["target_etf"] == "510300"].iloc[0]
    assert bool(row["file_exists"]) is True
    assert bool(row["header_only"]) is True
    assert row["diagnosis"] == "header_only_empty_export"


def test_target_etf_file_missing_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path, monkeypatch)

    frame = rescue.diagnose_empty_exports(config.export_roots)

    assert "159915" in set(frame["target_etf"])
    assert (frame["diagnosis"] == "target_etf_file_missing").any()


def test_forbidden_account_trade_filename_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    (root / "510300_account_trade_fill.lc5").write_bytes(b"blocked")
    config_for(tmp_path, monkeypatch, roots=[root])

    inventory, candidates = rescue.inventory_local_data_files([root])

    assert bool(inventory.loc[0, "skipped_for_safety"]) is True
    assert inventory.loc[0, "status"] == "skipped_forbidden_path"
    assert candidates.empty


def test_local_candidate_inventory_detects_lc5_csv_txt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    (root / "510300.lc5").write_bytes(b"x" * 32)
    (root / "510050_5m.csv").write_text("x\n", encoding="utf-8")
    (root / "159915_5m.txt").write_text("x\n", encoding="utf-8")
    config_for(tmp_path, monkeypatch, roots=[root])

    _inventory, candidates = rescue.inventory_local_data_files([root])

    assert {Path(path).suffix for path in candidates["path"]} == {".lc5", ".csv", ".txt"}


def test_text_5m_file_can_be_normalized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    path = root / "510300_5m.csv"
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
                "amount": 1050,
            }
        ]
    ).to_csv(path, index=False)
    config_for(tmp_path, monkeypatch, roots=[root])

    frame, info = rescue.parse_candidate(path)

    assert info["parser"] == "text"
    assert frame.loc[0, "etf_code"] == "510300"
    assert frame.loc[0, "vwap"] == pytest.approx(1.05)


def test_unmappable_file_unsupported_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    path = root / "510300_5m.dat"
    path.write_bytes(b"not a supported local format")
    config_for(tmp_path, monkeypatch, roots=[root])

    frame, info = rescue.parse_candidate(path)

    assert frame.empty
    assert info["status"] == "unsupported_suffix"


def test_output_outside_local_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path, monkeypatch)
    bad = rescue.RescueConfig(
        local_roots=config.local_roots,
        export_roots=config.export_roots,
        raw_export_dir=config.raw_export_dir,
        out_dir=Path("docs/research/not_allowed"),
        rescued_dir=config.rescued_dir,
        run_packager=False,
    )

    with pytest.raises(rescue.BrokerClientLocalDataRescueError, match="path must be under .local_research_outputs"):
        rescue.rescue_local_data(bad)


def test_boundary_fields_all_disallowed_flags_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports"
    root.mkdir(parents=True, exist_ok=True)
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
            }
        ]
    ).to_csv(root / "510300_5m.csv", index=False)
    config = config_for(tmp_path, monkeypatch, roots=[root])

    decision = rescue.rescue_local_data(config)
    decision_path = tmp_path / ".local_research_outputs/aetfq3_lab/intraday_broker_client_local_data_rescue/rescue_decision.json"
    payload = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision["decision"] == "BROKER_CLIENT_LOCAL_5M_RESCUE_READY_FOR_MANUAL_INTAKE"
    for key in [
        "broker_login_used",
        "broker_network_used",
        "qmt_used",
        "xtdata_used",
        "account_api_used",
        "position_api_used",
        "order_api_used",
        "trade_api_used",
        "fill_api_used",
        "contains_account",
        "contains_position",
        "contains_order",
        "contains_trade",
        "contains_fill",
        "contains_secret",
        "model_training_allowed",
        "labels_generated",
        "order_intent_generated",
        "automatic_order_allowed",
        "stable_affected",
        "stable_promotion_ready",
        "metrics_are_effectiveness_evidence",
    ]:
        assert payload[key] is False
