"""TEMPORARY DIAGNOSTIC: summarize a trace; never change matching outputs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from party_matching.domain import read_jsonl


def clean(value: object) -> str:
    return " ".join(str(value or "").split()).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze --diagnose-rules candidate traces.")
    parser.add_argument("--run", required=True, type=Path, help="Output directory from a diagnostic run")
    parser.add_argument("--examples", type=int, default=5, help="Maximum held-out examples per failure bucket")
    args = parser.parse_args()
    trace_path = args.run / "rules_candidate_trace.jsonl"
    if not trace_path.exists():
        parser.error(f"{trace_path} is missing. Rerun scripts/run.py with --diagnose-rules.")

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)
    csv_path = args.run / "rules_diagnostics.csv"
    fields = [
        "split", "bucket", "raw_name", "expected_name", "decision", "reason",
        "decision_tier", "rules_plain_threshold",
        "expected_rank_all", "expected_rank_eligible", "expected_score", "expected_guard",
        "winner_name", "winner_score", "winner_guard", "score_gap",
        "candidate_count", "top_candidates", "adm_party_id", "expected_root_id",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in read_jsonl(trace_path):
            split = "CALIBRATION" if row["split"] == "CALIBRATION" else "TEST"
            bucket = row["bucket"]
            counters[split][bucket] += 1
            reasons[split][row["reason"]] += 1
            if split == "TEST" and len(examples[bucket]) < max(0, args.examples):
                examples[bucket].append(row)
            expected = row.get("expected_candidate") or {}
            winner = row.get("winner") or {}
            expected_score = expected.get("score")
            winner_score = winner.get("score")
            writer.writerow({
                "split": row["split"], "bucket": bucket, "raw_name": row["raw_name"],
                "expected_name": row.get("expected_name"), "decision": row["decision"],
                "reason": row["reason"], "decision_tier": row.get("decision_tier"),
                "rules_plain_threshold": row.get("rules_plain_threshold"),
                "expected_rank_all": row.get("expected_rank_all"),
                "expected_rank_eligible": row.get("expected_rank_eligible"),
                "expected_score": expected_score, "expected_guard": expected.get("guard"),
                "winner_name": winner.get("root_name"), "winner_score": winner_score,
                "winner_guard": winner.get("guard"),
                "score_gap": round(winner_score - expected_score, 6)
                if winner_score is not None and expected_score is not None else None,
                "candidate_count": row.get("candidate_count"),
                "top_candidates": " | ".join(
                    f"{item['root_name']} ({item['score']:.3f}{', guard=' + item['guard'] if item['guard'] else ''})"
                    for item in row.get("top_candidates", [])
                ),
                "adm_party_id": row["adm_party_id"],
                "expected_root_id": row["expected_root_id"],
            })

    md_path = args.run / "rules_diagnostics.md"
    lines = [
        "# Rules-only candidate diagnosis",
        "",
        "This report covers missed, labeled, matchable plain-name rows only. It is a read-only",
        "audit of the existing rules decisions, not a proposal to lower any cutoff. OBO/VIA and",
        "unknown rows are intentionally excluded. Candidate rank is by final rules target score.",
        "",
        "| Failure bucket | Calibration | Held-out test |",
        "|---|---:|---:|",
    ]
    for bucket in sorted(set(counters["CALIBRATION"]) | set(counters["TEST"])):
        lines.append(f"| {bucket} | {counters['CALIBRATION'][bucket]:,} | {counters['TEST'][bucket]:,} |")
    lines += [
        f"| **Total traced misses** | **{sum(counters['CALIBRATION'].values()):,}** | **{sum(counters['TEST'].values()):,}** |",
        "",
        "Bucket order is deliberate: missing retrieval, guard rejection, lower eligible rank,",
        "ambiguity, then insufficient support. A row can have multiple contributing factors;",
        "the CSV retains its candidate rank, scores, and guard for closer inspection.",
        "",
        "## Final decision reasons",
        "",
        "| Reason | Calibration | Held-out test |",
        "|---|---:|---:|",
    ]
    for reason in sorted(set(reasons["CALIBRATION"]) | set(reasons["TEST"])):
        lines.append(f"| {reason} | {reasons['CALIBRATION'][reason]:,} | {reasons['TEST'][reason]:,} |")
    for bucket in sorted(examples):
        lines += [
            "", f"## {bucket}: held-out examples", "",
            "| Raw name | Expected | Winner | Expected rank | Expected score | Winner score | Reason |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for row in examples[bucket]:
            expected = row.get("expected_candidate") or {}
            winner = row.get("winner") or {}
            lines.append(
                f"| {clean(row['raw_name'])} | {clean(row.get('expected_name'))} | "
                f"{clean(winner.get('root_name'))} | {row.get('expected_rank_all') or '—'} | "
                f"{expected.get('score', '—')} | {winner.get('score', '—')} | {row['reason']} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path} and {csv_path}")
    print(f"Traced {sum(counters['CALIBRATION'].values()):,} calibration and "
          f"{sum(counters['TEST'].values()):,} held-out plain-name misses")
    for bucket, count in counters["TEST"].most_common():
        print(f"  {bucket:<22} {count:>7,}")


if __name__ == "__main__":
    main()
