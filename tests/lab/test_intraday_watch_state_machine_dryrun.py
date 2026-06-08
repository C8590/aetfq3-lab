from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_watch_state_machine_dryrun import (
    EVENTS,
    STATES,
    run_intraday_watch_dryrun,
    transition_state,
)


EVENTS_FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_watch_events.json"


def test_all_required_states_and_events_defined():
    assert STATES == {
        "WAIT_OPEN",
        "OPEN_GAP_TOO_HIGH",
        "INTRADAY_CONFIRMING",
        "PROBE_READY",
        "WAIT_PULLBACK",
        "CANCEL_BUY",
        "HOLD_LOCKED",
        "PROFIT_PROTECT",
        "EXIT_READY",
    }
    assert EVENTS == {
        "OPEN_PRINTED",
        "GAP_TOO_HIGH",
        "OPEN_RANGE_CONFIRMED",
        "VWAP_PULLBACK",
        "VWAP_RECLAIM",
        "BREAK_OPEN_LOW",
        "SECTOR_CONFIRM",
        "SECTOR_FADE",
        "RISK_CANCEL",
        "LOCK_HOLD_START",
        "PROFIT_PROTECT_TRIGGER",
        "EXIT_SIGNAL_RESEARCH_ONLY",
    }


def test_transition_examples_cover_cancel_and_gap():
    assert transition_state("WAIT_OPEN", "GAP_TOO_HIGH") == "OPEN_GAP_TOO_HIGH"
    assert transition_state("PROBE_READY", "BREAK_OPEN_LOW") == "CANCEL_BUY"
    assert transition_state("WAIT_PULLBACK", "RISK_CANCEL") == "CANCEL_BUY"


def test_state_machine_dryrun_visited_expected_states(tmp_path: Path):
    out_dir = tmp_path / "dryrun"
    report = run_intraday_watch_dryrun(EVENTS_FIXTURE, out_dir)

    assert report["status"] == "passed"
    assert report["visited_states"] == [
        "WAIT_OPEN",
        "INTRADAY_CONFIRMING",
        "PROBE_READY",
        "WAIT_PULLBACK",
        "HOLD_LOCKED",
        "PROFIT_PROTECT",
        "EXIT_READY",
    ]
    assert report["event_count"] == 8
    assert report["terminal_state"] == "EXIT_READY"
    assert (out_dir / "intraday_watch_dryrun_report.json").exists()
    assert (out_dir / "intraday_watch_dryrun_report.md").exists()


def test_dryrun_does_not_generate_order_intent(tmp_path: Path):
    report = run_intraday_watch_dryrun(EVENTS_FIXTURE, tmp_path)

    assert report["order_intent_generated"] is False
    assert report["does_not_generate_order_intent"] is True
    assert not (tmp_path / "OrderIntent.json").exists()
    assert not (tmp_path / "intraday_watch_snapshot.json").exists()
    assert not (tmp_path / "intraday_watch_events.csv").exists()


def test_dryrun_qmt_and_advisory_boundaries(tmp_path: Path):
    report = run_intraday_watch_dryrun(EVENTS_FIXTURE, tmp_path)

    assert report["qmt_allowed"] is False
    assert report["advisory_only"] is True
    assert report["stable_effect_allowed"] is False

