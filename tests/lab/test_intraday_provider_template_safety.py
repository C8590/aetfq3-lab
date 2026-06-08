from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_5m_market_data_adapter import (
    IntradayMarketDataAdapterError,
    export_intraday_5m_bars,
    parse_date,
)
from tools.lab.intraday_market_data_safety_scan import scan_path
from tools.lab.intraday_provider_template import (
    Intraday5mProviderTemplate,
    REAL_PROVIDER_TEMPLATE_ERROR,
)


TEMPLATE = REPO_ROOT / "tools/lab/intraday_provider_template.py"
SAFE_TEMPLATE_FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_provider_template_safe.py"
UNSAFE_PROVIDER = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_provider_unsafe.py"


def test_provider_template_imports():
    assert Intraday5mProviderTemplate.__name__ == "Intraday5mProviderTemplate"


def test_provider_capabilities_are_market_data_only():
    capabilities = Intraday5mProviderTemplate.provider_capabilities()

    assert capabilities == {
        "market_data_only": True,
        "supports_account": False,
        "supports_position": False,
        "supports_order": False,
        "supports_trade": False,
        "supports_submit_order": False,
        "supports_cancel_order": False,
        "supports_order_intent": False,
        "requires_secret": False,
        "requires_live_session": False,
    }


def test_real_provider_template_call_requires_separate_review():
    with pytest.raises(NotImplementedError, match=REAL_PROVIDER_TEMPLATE_ERROR):
        Intraday5mProviderTemplate().get_5m_bars(
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
        )


def test_safety_scan_allows_provider_template():
    result = scan_path(TEMPLATE)

    assert result["safe"] is True
    assert result["severity"] == "none"
    assert result["forbidden_hits"] == []


def test_safety_scan_allows_mock_template_fixture():
    result = scan_path(SAFE_TEMPLATE_FIXTURE)

    assert result["safe"] is True
    assert result["severity"] == "none"
    assert result["forbidden_hits"] == []


def test_existing_unsafe_fixture_still_fails_with_traceable_hits():
    result = scan_path(UNSAFE_PROVIDER)
    hit_keywords = {hit["keyword"] for hit in result["forbidden_hits"]}

    assert result["safe"] is False
    assert result["severity"] == "P0"
    assert {"submit_order", "get_account", "cancel_order", "get_positions", "order_intent"}.issubset(hit_keywords)
    assert all(hit["path"].endswith("mock_intraday_provider_unsafe.py") for hit in result["forbidden_hits"])
    assert all(hit["severity"] == "P0" for hit in result["forbidden_hits"])


@pytest.mark.parametrize("provider", ["qmt", "xtdata", "live"])
def test_adapter_still_blocks_real_provider_names(provider: str, tmp_path: Path):
    with pytest.raises(IntradayMarketDataAdapterError, match="real provider not implemented"):
        export_intraday_5m_bars(
            provider=provider,
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
            out_dir=tmp_path / ".local_research_outputs" / "blocked",
        )


def test_output_directory_and_execution_intent_boundaries(tmp_path: Path):
    with pytest.raises(IntradayMarketDataAdapterError, match="must not be output"):
        export_intraday_5m_bars(
            provider="mock",
            symbols=["510300"],
            start_date=parse_date("2026-06-01"),
            end_date=parse_date("2026-06-01"),
            out_dir=tmp_path / "output" / ".local_research_outputs" / "blocked",
        )

    assert "OrderIntent" not in TEMPLATE.read_text(encoding="utf-8")
