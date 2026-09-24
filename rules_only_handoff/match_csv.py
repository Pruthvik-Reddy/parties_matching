"""CLI: python match_csv.py --verified verified.csv --unverified unverified.csv --output predictions.csv"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from party_matching_rules import match_parties


OUTPUT_COLUMNS = [
    "unverified_id", "unverified_name", "decision", "verified_id",
    "verified_name", "score", "reason", "decision_tier", "matched_mention",
]


def read_rows(path: Path, name_column: str, id_column: str, prefix: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or name_column not in reader.fieldnames:
            raise ValueError(f"{path} needs a {name_column} column")
        rows = []
        for index, row in enumerate(reader, start=1):
            if not str(row.get(name_column) or "").strip():
                raise ValueError(f"Blank {name_column} on CSV data row {index} in {path}")
            row[id_column] = str(row.get(id_column) or f"{prefix}{index}").strip()
            rows.append(row)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone enhanced rules-only party matcher.")
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--unverified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.verified.resolve(), args.unverified.resolve()}:
        parser.error("Output must not overwrite either input CSV")
    verified = read_rows(args.verified, "verified_name", "verified_id", "v")
    unverified = read_rows(args.unverified, "unverified_name", "unverified_id", "u")
    predictions = match_parties(verified, unverified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(predictions)
    matched = sum(row["decision"] == "MATCH" for row in predictions)
    print(f"Wrote {args.output.resolve()} ({len(predictions):,} rows, {matched:,} matches)")


if __name__ == "__main__":
    main()
