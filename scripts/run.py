from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from party_matching.domain import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run party matching and create events plus predictions.xlsx.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--mode", choices=("serial", "parallel"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", help="Optional text file with one ADM party ID per line.")
    parser.add_argument("--output")
    parser.add_argument("--fresh-state", action="store_true", help="Use isolated state for a reproducible evaluation run.")
    args = parser.parse_args()
    config = load_config(args.config)
    native_threads = str(config.get("execution", {}).get("native_threads", 10))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, native_threads)
    from party_matching.matching import run_matching
    from party_matching.reporting import build_reports
    if args.mode:
        config.setdefault("execution", {})["mode"] = args.mode
    if args.limit:
        config.setdefault("execution", {})["limit"] = args.limit
    if args.ids_file:
        config.setdefault("execution", {})["ids_file"] = args.ids_file
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = Path(args.output or Path(config.get("paths", {}).get("outputs_dir", "outputs")) / run_id)
    stats = run_matching(config, output, fresh_state=args.fresh_state)
    metrics = build_reports(config.get("paths", {}).get("prepared_dir", "data/prepared"), output, stats)
    overall = metrics["overall"]
    precision = "n.a." if overall["precision"] is None else f"{overall['precision']:.2%}"
    print(f"Wrote {output.resolve()}")
    print(f"Rows={overall['rows']:,} matches={overall['matches']:,} precision={precision} runtime={stats['total_seconds']:.2f}s")


if __name__ == "__main__":
    main()
