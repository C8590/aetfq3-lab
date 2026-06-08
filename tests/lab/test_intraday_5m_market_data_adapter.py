from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_5m_market_data_adapter import (
    IntradayMarketDataAdapterError,
    MockIntraday5mProvider,
    export_intraday_5m_bars,
    parse_date,
    validate_intraday_5m_frame,
)


def test_mock_provider_can_export_5m_csv(tmp_path: Path):
    out_dir = tmp_path / ".local_research_outputs" / "aetfq3_lab" / "adapter_smoke"

    result = export_intraday_5m_bars(
        provider="mock",
        symbols=["510300", "159915"],
        start_date=parse_date("2026-06-01"),
        end_date=parse_date("2026-06-02"),
        out_dir=out_dir,
    )

    csv_path = out_dir / "mock_intraday_5m_export.csv"
    manifest_path = out_dir / "EXPORT_MANIFEST.json"
    frame = pd.read_csv(csv_path, dtype={"etf_code": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result["status"] == "exported"
    assert csv_path.exists()
    assert manifest_path.exists()
    assert (out_dir / "SHA256SUMS.txt").exists()
    assert (out_dir / "source_note.md").exists()
    assert set(frame["etf_code"]) == {"510300", "159915"}
    assert manifest["data_is_mock"] is True
    assert manifest["stable_allowed"] is False
    assert manifest["real_provider_enabled"] is False


def test_output_path_must_be_ignored_local_directory(tmp_path: Path):
    with pytest.raises(IntradayMarketDataAdapterError, match="ignored local directory"):
        export_intraday_5m_bars(
            provider="mock",
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
            out_dir=tmp_path / "plain_dir",
        )


@pytest.mark.parametrize("provider", ["qmt", "xtdata", "live"])
def test_real_providers_must_fail(provider: str, tmp_path: Path):
    with pytest.raises(IntradayMarketDataAdapterError, match="real provider not implemented"):
        export_intraday_5m_bars(
            provider=provider,
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
            out_dir=tmp_path / ".local_research_outputs" / "blocked",
        )


def test_validate_intraday_5m_frame_requires_columns():
    frame = MockIntraday5mProvider().get_5m_bars(["510300"], parse_date("2026-06-01"), parse_date("2026-06-01"))
    validate_intraday_5m_frame(frame)

    with pytest.raises(IntradayMarketDataAdapterError, match="missing required columns"):
        validate_intraday_5m_frame(frame.drop(columns=["amount"]))


def test_adapter_outputs_do_not_create_execution_intent_or_output_dir(tmp_path: Path):
    out_dir = tmp_path / ".local_artifact_backup" / "adapter_smoke"
    export_intraday_5m_bars(
        provider="mock",
        symbols=["510300"],
        start_date=parse_date("2026-06-01"),
        end_date=parse_date("2026-06-01"),
        out_dir=out_dir,
    )

    file_names = {path.name for path in out_dir.iterdir()}
    combined_text = "\n".join(path.read_text(encoding="utf-8") for path in out_dir.iterdir() if path.is_file())
    assert not any(name.endswith(".intent") for name in file_names)
    assert "OrderIntent" not in combined_text
    assert "real_provider_enabled\": false" in combined_text

    with pytest.raises(IntradayMarketDataAdapterError, match="must not be output"):
        export_intraday_5m_bars(
            provider="mock",
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
            out_dir=tmp_path / "output" / ".local_research_outputs" / "blocked",
        )
