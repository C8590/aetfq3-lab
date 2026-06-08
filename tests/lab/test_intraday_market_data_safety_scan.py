from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_market_data_safety_scan import scan_path


SAFE_PROVIDER = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_provider_safe.py"
UNSAFE_PROVIDER = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_provider_unsafe.py"
ADAPTER = REPO_ROOT / "tools/lab/intraday_5m_market_data_adapter.py"


def test_safety_scanner_allows_safe_provider():
    result = scan_path(SAFE_PROVIDER)

    assert result["safe"] is True
    assert result["forbidden_hits"] == []
    assert result["p0_blockers"] == []


def test_safety_scanner_blocks_unsafe_provider():
    result = scan_path(UNSAFE_PROVIDER)
    hit_keywords = {hit["keyword"] for hit in result["forbidden_hits"]}

    assert result["safe"] is False
    assert {"submit_order", "get_account", "cancel_order"}.issubset(hit_keywords)
    assert result["p0_blockers"]


def test_adapter_source_has_no_execution_api_hits():
    result = scan_path(ADAPTER)
    hit_keywords = {hit["keyword"] for hit in result["forbidden_hits"]}

    assert "submit_order" not in hit_keywords
    assert "cancel_order" not in hit_keywords
    assert "get_account" not in hit_keywords
    assert "get_positions" not in hit_keywords
