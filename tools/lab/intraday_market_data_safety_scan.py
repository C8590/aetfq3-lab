from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FORBIDDEN_KEYWORDS = [
    "submit_order",
    "cancel_order",
    "order_stock",
    "passorder",
    "buy",
    "sell",
    "get_position",
    "get_positions",
    "get_asset",
    "get_assets",
    "get_account",
    "account",
    "position",
    "positions",
    "order",
    "orders",
    "trade",
    "trades",
    "deal",
    "entrust",
    "fund",
    "balance",
    "cash",
    "order_intent",
    "OrderIntent",
    "live_order",
    "secret",
    "token",
    "password",
    "api_key",
]


@dataclass(frozen=True)
class ForbiddenHit:
    keyword: str
    line: int
    text: str

    def as_dict(self) -> dict[str, object]:
        return {"keyword": self.keyword, "line": self.line, "text": self.text}


def scan_text(text: str) -> list[ForbiddenHit]:
    hits: list[ForbiddenHit] = []
    patterns = [(keyword, _keyword_pattern(keyword)) for keyword in FORBIDDEN_KEYWORDS]
    for line_number, line in enumerate(text.splitlines(), start=1):
        for keyword, pattern in patterns:
            if pattern.search(line):
                hits.append(ForbiddenHit(keyword=keyword, line=line_number, text=line.strip()))
    return hits


def scan_path(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    text = resolved.read_text(encoding="utf-8")
    hits = scan_text(text)
    p0_blockers = [
        f"forbidden keyword '{hit.keyword}' found at line {hit.line}" for hit in hits
    ]
    return {
        "safe": not hits,
        "forbidden_hits": [hit.as_dict() for hit in hits],
        "path": str(resolved),
        "scan_scope": "single_file_static_keyword_scan",
        "p0_blockers": p0_blockers,
    }


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    flags = 0 if any(character.isupper() for character in keyword) else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(keyword)}(?![A-Za-z0-9_])", flags)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab intraday market-data static safety scanner")
    parser.add_argument("--path", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = scan_path(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
