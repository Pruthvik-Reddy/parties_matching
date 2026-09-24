from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from party_matching.domain import load_config, write_json


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run party matching and create events plus predictions.xlsx.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--mode", choices=("serial", "parallel"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", help="Optional text file with one ADM party ID per line.")
    parser.add_argument("--output")
    parser.add_argument("--fresh-state", action="store_true", help="Use isolated state for a reproducible evaluation run.")
    # TEMPORARY DIAGNOSTIC: remove together with the hook in run_matching.
    parser.add_argument("--diagnose-rules", action="store_true", help="Write an opt-in, read-only candidate trace for missed labeled plain-name rules decisions.")
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
    stats = run_matching(
        config, output, fresh_state=args.fresh_state,
        diagnose_rules=args.diagnose_rules,
    )
    report_started = time.perf_counter()
    metrics = build_reports(config.get("paths", {}).get("prepared_dir", "data/prepared"), output, stats)
    stats["report_seconds"] = time.perf_counter() - report_started
    stats["end_to_end_seconds"] = time.perf_counter() - started
    try:
        import psutil
        memory = psutil.Process().memory_info()
        stats["post_report_rss_mb"] = memory.rss / (1024 * 1024)
        stats["peak_rss_mb"] = max(stats["peak_rss_mb"] or 0, getattr(memory, "peak_wset", memory.rss) / (1024 * 1024))
    except ImportError:
        pass
    write_json(output / "run_stats.json", stats)
    metrics["runtime"] = stats
    write_json(output / "run_metrics.json", metrics)
    with (output / "analysis.md").open("a", encoding="utf-8") as analysis:
        analysis.write(f"\n## End-to-end profile\n"
                       f"\n- Report generation: {stats['report_seconds']:.2f} seconds"
                       f"\n- End-to-end runtime: {stats['end_to_end_seconds']:.2f} seconds\n")
    overall = metrics["overall"]
    precision = "n.a." if overall["precision"] is None else f"{overall['precision']:.2%}"
    recall = "n.a." if overall["recall"] is None else f"{overall['recall']:.2%}"
    print(f"Wrote {output.resolve()}")
    print(
        f"Rows={overall['rows']:,} matches={overall['matches']:,} "
        f"precision={precision} recall={recall} runtime={stats['end_to_end_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
