from __future__ import annotations

import argparse
from pathlib import Path

from party_matching.domain import load_config, read_json
from party_matching.reporting import build_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a rules-only demo workbook from an existing completed run."
    )
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--run", required=True, help="Existing output folder containing rules_decisions.jsonl.")
    args = parser.parse_args()

    output = Path(args.run)
    for filename in ("decisions.jsonl", "rules_decisions.jsonl", "run_stats.json"):
        if not (output / filename).is_file():
            parser.error(f"Required run file is missing: {output / filename}")
    config = load_config(args.config)
    prepared = Path(config.get("paths", {}).get("prepared_dir", "data/prepared"))
    for filename in ("manifest.json", "labels.jsonl", "source_rows.jsonl"):
        if not (prepared / filename).is_file():
            parser.error(f"Required prepared-data file is missing: {prepared / filename}")

    stats = read_json(output / "run_stats.json", {}) or {}
    metrics = build_reports(prepared, output, stats, rules_demo_only=True)
    overall = metrics["overall"]
    precision = "n.a." if overall["precision"] is None else f"{overall['precision']:.1%}"
    recall = "n.a." if overall["recall"] is None else f"{overall['recall']:.1%}"
    print(f"Wrote {(output / 'predictions_rules_only.xlsx').resolve()}")
    print(f"Enhanced rules: {overall['matches']:,} matches, {overall['correct_matches']:,} correct, "
          f"precision {precision}, recall {recall}")


if __name__ == "__main__":
    main()
