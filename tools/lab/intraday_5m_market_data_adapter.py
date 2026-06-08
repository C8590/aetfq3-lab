from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REAL_PROVIDER_ERROR = "real provider not implemented; requires separate human authorization and safety review"
OUTPUT_CSV_NAME = "mock_intraday_5m_export.csv"
MANIFEST_NAME = "EXPORT_MANIFEST.json"
HASH_NAME = "SHA256SUMS.txt"
SOURCE_NOTE_NAME = "source_note.md"
REQUIRED_COLUMNS = [
    "trade_date",
    "datetime",
    "etf_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
OPTIONAL_COLUMNS = [
    "vwap",
    "etf_name",
    "sector",
    "provider",
    "source_kind",
]
ALLOWED_OUTPUT_ROOT_NAMES = {".local_research_outputs", ".local_artifact_backup"}


class IntradayMarketDataAdapterError(RuntimeError):
    pass


class MarketDataProviderProtocol(Protocol):
    def get_5m_bars(self, symbols: Sequence[str], start_date: date, end_date: date) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class MockIntraday5mProvider:
    provider_name: str = "mock"
    source_kind: str = "mock_intraday_5m_bar"

    def get_5m_bars(self, symbols: Sequence[str], start_date: date, end_date: date) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for current_date in _iter_dates(start_date, end_date):
            if current_date.weekday() >= 5:
                continue
            for symbol_index, symbol in enumerate(symbols):
                rows.extend(self._bars_for_symbol_day(symbol.strip(), symbol_index, current_date))
        return pd.DataFrame(rows)

    def _bars_for_symbol_day(self, symbol: str, symbol_index: int, current_date: date) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        base_price = 3.0 + symbol_index * 0.25
        for bar_index, bar_dt in enumerate(_a_share_5m_bar_datetimes(current_date)):
            drift = bar_index * 0.002
            open_price = round(base_price + drift, 4)
            close_price = round(open_price + 0.004, 4)
            high_price = round(max(open_price, close_price) + 0.006, 4)
            low_price = round(min(open_price, close_price) - 0.006, 4)
            volume = 100_000 + symbol_index * 10_000 + bar_index * 125
            amount = round(volume * close_price, 2)
            rows.append(
                {
                    "trade_date": current_date.isoformat(),
                    "datetime": bar_dt.isoformat(sep=" "),
                    "etf_code": symbol,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "amount": amount,
                    "vwap": round(amount / volume, 6),
                    "etf_name": f"MOCK-{symbol}",
                    "sector": "mock_sector",
                    "provider": self.provider_name,
                    "source_kind": self.source_kind,
                }
            )
        return rows


def validate_intraday_5m_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise IntradayMarketDataAdapterError("intraday 5m frame is empty")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise IntradayMarketDataAdapterError(f"missing required columns: {missing}")

    if df["etf_code"].astype(str).str.strip().eq("").any():
        raise IntradayMarketDataAdapterError("etf_code contains blank values")

    pd.to_datetime(df["trade_date"], errors="raise")
    pd.to_datetime(df["datetime"], errors="raise")

    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        numeric = pd.to_numeric(df[column], errors="raise")
        if numeric.isna().any():
            raise IntradayMarketDataAdapterError(f"{column} contains null numeric values")

    invalid_price_rows = df[
        (pd.to_numeric(df["high"]) < pd.to_numeric(df["low"]))
        | (pd.to_numeric(df["high"]) < pd.to_numeric(df["open"]))
        | (pd.to_numeric(df["high"]) < pd.to_numeric(df["close"]))
        | (pd.to_numeric(df["low"]) > pd.to_numeric(df["open"]))
        | (pd.to_numeric(df["low"]) > pd.to_numeric(df["close"]))
    ]
    if not invalid_price_rows.empty:
        raise IntradayMarketDataAdapterError("OHLC bounds are invalid")


def export_intraday_5m_bars(
    provider: str,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    out_dir: Path,
) -> dict[str, object]:
    normalized_provider = provider.strip().lower()
    if normalized_provider != "mock":
        raise IntradayMarketDataAdapterError(REAL_PROVIDER_ERROR)

    clean_symbols = [symbol.strip() for symbol in symbols if symbol.strip()]
    if not clean_symbols:
        raise IntradayMarketDataAdapterError("symbols must not be empty")
    if start_date > end_date:
        raise IntradayMarketDataAdapterError("start_date must be on or before end_date")

    out_dir = ensure_ignored_out_dir(out_dir)
    provider_instance = MockIntraday5mProvider()
    frame = provider_instance.get_5m_bars(clean_symbols, start_date, end_date)
    validate_intraday_5m_frame(frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / OUTPUT_CSV_NAME
    manifest_path = out_dir / MANIFEST_NAME
    hash_path = out_dir / HASH_NAME
    source_note_path = out_dir / SOURCE_NOTE_NAME

    ordered_columns = [*REQUIRED_COLUMNS, *[column for column in OPTIONAL_COLUMNS if column in frame.columns]]
    frame = frame[ordered_columns].sort_values(["trade_date", "etf_code", "datetime"])
    frame.to_csv(csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    file_hashes = {csv_path.name: _sha256_file(csv_path)}
    manifest = _build_manifest(frame, provider=normalized_provider, symbols=clean_symbols, start_date=start_date, end_date=end_date)
    manifest["files"] = [{"name": name, "sha256": digest} for name, digest in file_hashes.items()]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    file_hashes[manifest_path.name] = _sha256_file(manifest_path)

    source_note_path.write_text(_source_note(), encoding="utf-8")
    file_hashes[source_note_path.name] = _sha256_file(source_note_path)
    hash_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(file_hashes.items())),
        encoding="utf-8",
    )

    return {
        "status": "exported",
        "provider": normalized_provider,
        "out_dir": str(out_dir),
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "row_count": int(len(frame)),
        "symbols": clean_symbols,
        "files": [csv_path.name, manifest_path.name, hash_path.name, source_note_path.name],
    }


def ensure_ignored_out_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    if "output" in {part.lower() for part in resolved.parts}:
        raise IntradayMarketDataAdapterError("out_dir must not be output/")
    if not any(part in ALLOWED_OUTPUT_ROOT_NAMES for part in resolved.parts):
        allowed = ", ".join(sorted(ALLOWED_OUTPUT_ROOT_NAMES))
        raise IntradayMarketDataAdapterError(f"out_dir must be under an ignored local directory: {allowed}")
    return resolved


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _iter_dates(start_date: date, end_date: date) -> list[date]:
    days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def _a_share_5m_bar_datetimes(current_date: date) -> list[datetime]:
    morning = _time_range(current_date, time(9, 35), time(11, 30), 5)
    afternoon = _time_range(current_date, time(13, 5), time(15, 0), 5)
    return [*morning, *afternoon]


def _time_range(current_date: date, start: time, end: time, step_minutes: int) -> list[datetime]:
    cursor = datetime.combine(current_date, start)
    stop = datetime.combine(current_date, end)
    values: list[datetime] = []
    while cursor <= stop:
        values.append(cursor)
        cursor += timedelta(minutes=step_minutes)
    return values


def _build_manifest(
    frame: pd.DataFrame,
    provider: str,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
) -> dict[str, object]:
    return {
        "lab_declaration": LAB_DECLARATION,
        "phase": "F-5 Lab-only intraday 5m market-data adapter smoke",
        "provider": provider,
        "real_provider_enabled": False,
        "stable_allowed": False,
        "qmt_trade_allowed": False,
        "execution_intent_allowed": False,
        "contains_secret": False,
        "contains_live_execution": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbols),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "row_count": int(len(frame)),
        "etf_count": int(frame["etf_code"].nunique()),
        "trade_date_count": int(frame["trade_date"].nunique()),
        "columns": list(frame.columns),
        "output_policy": "ignored local directory only",
        "data_is_mock": True,
        "not_real_market_data": True,
    }


def _source_note() -> str:
    return "\n".join(
        [
            LAB_DECLARATION,
            "This smoke export is mock market-data only.",
            "It does not connect to QMT, Stable, broker APIs, or execution systems.",
            "It must not be interpreted as real market data or a formal trading plan.",
            "",
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-only intraday 5m market-data adapter")
    parser.add_argument("--provider", required=True, help="Only 'mock' is currently supported")
    parser.add_argument("--symbols", required=True, help="Comma-separated ETF codes")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = export_intraday_5m_bars(
            provider=args.provider,
            symbols=args.symbols.split(","),
            start_date=parse_date(args.start_date),
            end_date=parse_date(args.end_date),
            out_dir=args.out_dir,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
