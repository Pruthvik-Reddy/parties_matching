"""Read-only threshold audit for plain-name rules decisions from an existing run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rate(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rules-only plain-name recall without rerunning matching.")
    parser.add_argument("--run", required=True, type=Path, help="Existing run output directory")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--margin", type=float, default=0.04, help="Must match matching.minimum_root_margin")
    args = parser.parse_args()

    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((args.run / "run_stats.json").read_text(encoding="utf-8"))
    data_version = manifest.get("data_version")
    model_version = str(stats.get("model_version") or "")
    if data_version and model_version and data_version not in model_version:
        parser.error("The run and prepared data have different versions; pass the prepared directory used for that run.")

    labels = {
        row["adm_party_id"]: row
        for row in rows(args.prepared / "labels.jsonl")
        if row.get("scorable") and row.get("split") in {"CALIBRATION", "TEST_KNOWN", "TEST_UNSEEN"}
    }
    groups = defaultdict(lambda: {"rows": 0, "matches": 0, "correct": 0, "rejected": [], "reasons": Counter()})
    for decision in rows(args.run / "rules_decisions.jsonl"):
        label = labels.get(decision["adm_party_id"])
        if not label or decision.get("connector_resolution") is not None or decision.get("parse_warning"):
            continue
        split = "CALIBRATION" if label["split"] == "CALIBRATION" else "TEST"
        group = groups[split]
        group["rows"] += 1
        correct = decision.get("verified_party_id") == label["expected_party_id"]
        if decision["decision"] == "MATCH":
            group["matches"] += 1
            group["correct"] += int(correct)
        else:
            retrieved = label["expected_party_id"] in decision.get("retrieved_root_ids", [])
            group["reasons"][(decision["reason"], "retrieved" if retrieved else "missed")] += 1
            # Only these rows were rejected solely by the confidence cutoff.
            if (
                decision["reason"] == "INSUFFICIENT_SUPPORT"
                and decision.get("verified_party_id")
                and decision.get("margin") is not None
                and decision["margin"] >= args.margin
                and not decision.get("digit_conflict")
                and not decision.get("distinctive_token_conflict")
            ):
                group["rejected"].append((float(decision["confidence"]), correct))

    for split in ("CALIBRATION", "TEST"):
        group = groups[split]
        total, matches, correct = group["rows"], group["matches"], group["correct"]
        print(f"\n{split}: {total:,} plain labeled rows")
        print(f"Current rules: {matches:,} matches, {correct:,} correct, "
              f"precision {rate(correct, matches)}, recall {rate(correct, total)}")
        print("Potential fuzzy cutoff | Added | Added correct | Added precision | Total precision | Total recall")
        for cutoff in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60):
            added = [is_correct for score, is_correct in group["rejected"] if score >= cutoff]
            extra = sum(added)
            print(f"{cutoff:>21.2f} | {len(added):>5,} | {extra:>13,} | "
                  f"{rate(extra, len(added)):>15} | {rate(correct + extra, matches + len(added)):>15} | "
                  f"{rate(correct + extra, total):>12}")
        print("Most common remaining NO_MATCH reasons (expected root retrieved/missed):")
        for (reason, retrieval), count in group["reasons"].most_common(10):
            print(f"  {reason:<28} {retrieval:<9} {count:>7,}")


if __name__ == "__main__":
    main()
