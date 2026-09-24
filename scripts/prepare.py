from __future__ import annotations

import argparse
from pathlib import Path

from party_matching.domain import load_config
from party_matching.prepare import create_sample_workbook, prepare_workbook


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe local ADM and evaluation files.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--workbook")
    parser.add_argument("--output")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    workbook = Path(args.workbook) if args.workbook else None
    if args.sample:
        workbook = create_sample_workbook("sample/related-parties.xlsx")
    if workbook is None:
        parser.error("provide --workbook or --sample")
    output = Path(args.output or config.get("paths", {}).get("prepared_dir", "data/prepared"))
    manifest = prepare_workbook(workbook, output, config)
    print(
        f"Prepared {manifest['cleaned_unverified_rows']:,} unverified rows from "
        f"{manifest['rows']:,} source rows and {manifest['verified_parties']:,} verified parties "
        f"in {output.resolve()}"
    )


if __name__ == "__main__":
    main()
