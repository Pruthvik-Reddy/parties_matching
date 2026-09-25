"""Read-only, scoped suggestions from the POC's enhanced rules-only matcher.

This intentionally reuses the POC retrieval, rules scoring, decisions, and
recovery stages. It does not run ML decisions, emit events, update mappings, or
save the verified graph. Scope filtering selects *input rows*, never winners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# The POC package may be installed non-editably on the demo machine. Always
# import the rules from this checkout so the report and implementation agree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from party_matching.domain import PartyRecord, load_config, parse_mentions, read_json, read_jsonl
from party_matching.graph import VerifiedGraph
from party_matching.matching import (
    CrossEncoderReranker, FeatureScorer, MentionRetriever, _best_per_root,
    _guard_reason, collect_proposals, decide_records,
)
from party_matching.rules_anchor import recover_verified_name_anchors
from party_matching.rules_enhancement import enhance_rules_decisions
from party_matching.rules_multipart import choose_multipart_rules

from .engine import name_tokens


def _current_graph(prepared: Path, config: dict, account_id: str,
                   extra_names: list[str], catalog_override: list[dict] | None = None) -> VerifiedGraph:
    """Use only the current catalog, copying in-scope POC alias/parent evidence.

    Loading the existing graph directly could retain parties no longer in the
    current verified catalog, which would make a snapshot misleading.
    """
    catalog = (catalog_override if catalog_override is not None
               else read_json(prepared / "verified_parties.json", []) or [])
    existing_path = Path(config.get("paths", {}).get("state_dir", "state")) / account_id / "graph.json"
    graph = VerifiedGraph(account_id, existing_path, load_existing=False)
    graph.add_parties(catalog)
    for name in extra_names:
        name = name.strip()
        if not name:
            continue
        if any(node.party_name.casefold() == name.casefold() for node in graph.nodes.values()):
            continue
        incoming_id = "incoming-" + hashlib.sha256(name.casefold().encode()).hexdigest()[:16]
        graph.add_parties([{"partyId": incoming_id, "partyName": name}])
    if existing_path.exists():
        prior = VerifiedGraph(account_id, existing_path, load_existing=True)
        for party_id, node in graph.nodes.items():
            old = prior.nodes.get(party_id)
            if old is None:
                continue
            node.candidates = list(old.candidates)
            node.parent_id = old.parent_id if old.parent_id in graph.nodes else None
        graph._rebuild_indexes()
    return graph  # Never saved.


def _record_case(record: PartyRecord) -> str:
    """Classify inputs without changing the POC's connector parser or policy."""
    mentions = parse_mentions(record)
    if any(mention.parse_warning for mention in mentions):
        return "MALFORMED"
    connectors = {mention.connector_before for mention in mentions if mention.connector_before}
    return next(iter(connectors)) if len(connectors) == 1 else ("MIXED" if connectors else "PLAIN")


def _eligible_records(prepared: Path) -> tuple[list[PartyRecord], Counter[str]]:
    records: list[PartyRecord] = []
    counts: Counter[str] = Counter()
    for raw in read_jsonl(prepared / "adm_records.jsonl"):
        record = PartyRecord(**raw)
        if not record.eligible or record.is_verified:
            counts["ineligible_or_verified"] += 1
            continue
        if not record.raw_name.strip():
            counts["blank"] += 1
            continue
        # The POC scores OBO/VIA mention chains and returns a guarded NO_MATCH
        # for malformed/mixed chains. Keep that behavior here too.
        records.append(record)
    return records, counts


def _scope(records: list[PartyRecord], graph: VerifiedGraph, min_group_size: int,
           max_groups: int, focus_names: list[str]) -> tuple[list[PartyRecord], list[dict]]:
    """Fast leading-name scope only; all verified parties still compete later.

    A count here is *not* a known match count. It measures raw names sharing a
    leading normalized token with at least one verified name. This cheap pass
    lets a demo run focus on high-volume brands before building TF-IDF indexes.
    """
    if min_group_size == 0 and max_groups == 0 and not focus_names:
        return records, []  # Unscoped mode preserves the full plain-name population.
    verified_anchors = {tokens[0] for node in graph.nodes.values()
                        if (tokens := name_tokens(node.party_name))}
    focus = {tokens[0] for name in focus_names if (tokens := name_tokens(name))}
    if focus_names and not focus:
        raise ValueError("No usable --focus-verified name was supplied")
    if focus and not focus <= verified_anchors:
        raise ValueError("A focused name does not share a leading anchor with a current verified party")
    counts: Counter[str] = Counter()
    anchors: dict[str, str] = {}
    for record in records:
        tokens = name_tokens(record.raw_name)
        if tokens and tokens[0] in verified_anchors:
            anchors[record.adm_party_id] = tokens[0]
            counts[tokens[0]] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    selected_anchors = [anchor for anchor, count in ranked
                        if count >= min_group_size and (not focus or anchor in focus)]
    if max_groups:
        selected_anchors = selected_anchors[:max_groups]
    allowed = set(selected_anchors)
    selected = [record for record in records if anchors.get(record.adm_party_id) in allowed]
    return selected, [{"anchor": anchor, "raw_unverified_rows": counts[anchor]}
                      for anchor in selected_anchors]


def _apply_request_cutoff(decisions: list, party_job: dict, default_job: dict,
                          scorer: FeatureScorer, config: dict) -> None:
    floor = float(config.get("decision", {}).get("rules_containment_min_confidence", 0.40))
    for decision in decisions:
        job = party_job.get(str(decision.matched_member_id or decision.verified_party_id), default_job)
        if decision.decision_tier in {"RULES_UNIQUE_CONTAINMENT",
                                      "RULES_ROOT_UNIQUE_CONNECTOR_SHORT_NAME",
                                      "RULES_ROOT_UNIQUE_CONNECTOR_PREFIX"}:
            cutoff = min(float(scorer.plain_threshold), floor)
        elif decision.connector is None and decision.connector_resolution is None:
            cutoff = float(scorer.plain_threshold)
        else:
            cutoff = scorer.system_threshold
        if decision.decision == "MATCH" and decision.confidence < max(
                cutoff, float(job.get("confidenceCutoff", 0.0))):
            decision.decision = "NO_MATCH"
            decision.reason = "BELOW_REQUEST_CUTOFF"


def _rules_decisions(records: list[PartyRecord], graph: VerifiedGraph, config: dict,
                     prepared: Path, jobs: list[dict],
                     retriever: MentionRetriever | None = None,
                     scorer: FeatureScorer | None = None,
                     candidate_rows_only: bool = False,
                     extra_candidate_ids: set[str] | None = None) -> tuple[list, dict, FeatureScorer, dict]:
    artifact = Path(config.get("paths", {}).get("artifacts_dir", "artifacts")) / "matcher.joblib"
    # Inference may use an updated verified catalog. The trained ML model is
    # disabled below, so its prepared-data version must not force retraining.
    # Reuse only the existing POC IDF/rules cutoff when the artifact is present.
    scorer = scorer or FeatureScorer(artifact)
    scorer.model = scorer.target_model = scorer.calibrator = None
    scorer.system_threshold = float(scorer.artifact.get(
        "rules_system_threshold", config.get("decision", {}).get("fallback_system_threshold", 0.90)))
    scorer.plain_threshold = float(config.get("decision", {}).get(
        "rules_plain_threshold", scorer.system_threshold))
    if not 0 <= scorer.plain_threshold <= 1:
        raise ValueError("rules_plain_threshold must be between 0 and 1")
    execution = config.get("execution", {})
    workers = (1 if execution.get("mode") == "serial" else
               max(1, int(execution.get("workers", 10))))
    retrieval_started = time.perf_counter()
    retriever = retriever or MentionRetriever(records, config.get("retrieval", {}), workers=workers)
    proposals, retrieval_stats = collect_proposals(graph, retriever, scorer)
    if candidate_rows_only:
        # The one-party demo queries a shared full-population index. Only rows
        # with an indexed proposal need the expensive decision stages; absent
        # rows are still included in the separate label-retrieval audit.
        eligible_ids = set(proposals) | (extra_candidate_ids or set())
        records = [record for record in records if record.adm_party_id in eligible_ids]
    for record_proposals in proposals.values():
        for proposal in record_proposals:
            proposal.feature_score = proposal.rules_score
            proposal.cross_encoder_score = None
    reranker = CrossEncoderReranker("unused-rules-model", "off")
    baseline = decide_records(records, proposals, graph, scorer, reranker, config)
    party_job = {str(party["partyId"]): job for job in jobs
                 for party in job.get("verifiedParties", [])}
    default_job = jobs[0] if jobs else {"confidenceCutoff": 0.0}
    _apply_request_cutoff(baseline, party_job, default_job, scorer, config)
    decisions = baseline
    stages = {"baseline": sum(item.decision == "MATCH" for item in baseline)}
    stages["poc_artifact_loaded"] = bool(scorer.artifact)
    if not scorer.artifact:
        stages["artifact_warning"] = (
            "POC artifact unavailable; rules use fallback IDF/cutoff, so results may differ from the evaluated POC")
    if config.get("decision", {}).get("rules_enhanced_enabled", False):
        decisions, stats = enhance_rules_decisions(
            records, proposals, decisions, graph, retriever, scorer,
            config, party_job, default_job)
        stages["enhancement"] = {key: value for key, value in stats.items()
                                 if not key.startswith("_")}
    labels = prepared / "labels.jsonl"
    if config.get("decision", {}).get("rules_multipart_enabled", False) and labels.exists():
        decisions, stats = choose_multipart_rules(
            records, decisions, graph, scorer, config.get("retrieval", {}), workers,
            labels, party_job, default_job,
            float(config.get("decision", {}).get("rules_multipart_min_confidence", 0.80)))
        stages["multipart"] = stats
        stages["multipart_note"] = (
            "Position prior uses TRAIN/CALIBRATION labels, as in the POC; "
            "selected TEST labels do not enter decisions. A scoped run can learn a different prior.")
    elif config.get("decision", {}).get("rules_multipart_enabled", False):
        stages["multipart_note"] = "Skipped: no labels.jsonl for the POC's position prior"
    if config.get("decision", {}).get("rules_verified_anchor_enabled", False):
        decisions, stats = recover_verified_name_anchors(
            records, decisions, graph, proposals, retriever, scorer, party_job, default_job)
        stages["anchor"] = stats
    stages["retrieval_seconds"] = time.perf_counter() - retrieval_started
    return decisions, proposals, scorer, stages


def _review_candidates(record: PartyRecord, decision, proposals: list, config: dict,
                       score_floor: float, lexical_floor: float, max_candidates: int,
                       within_top: float = 0.10) -> list[dict]:
    if decision.decision != "NO_MATCH":
        return []
    case = _record_case(record)
    if case in {"MALFORMED", "MIXED"}:
        return []
    if case in {"OBO", "VIA"}:
        # Scores on different connector segments do not decide which party the
        # row is about. Review only the POC's preferred (leftmost) segment.
        preferred_id = parse_mentions(record)[0].mention_id
        proposals = [item for item in proposals if item.mention_id == preferred_id]
    roots = _best_per_root(proposals)
    eligible = [item for item in roots.values()
                if not _guard_reason(item, config.get("matching", {}))
                and max(item.char_tfidf_score, item.word_tfidf_score, item.exact) >= lexical_floor
                and item.rules_score >= score_floor]
    eligible.sort(key=lambda item: (-item.rules_score, item.root_party_id))
    if not eligible:
        return []
    best = eligible[0].rules_score
    # Keep close competitors visible instead of forcing one ambiguous party.
    selected = [item for item in eligible if best - item.rules_score <= within_top][:max_candidates]
    return [{
        "verified_party_id": item.root_party_id,
        "verified_party": item.root_party_name,
        "unverified_party_id": decision.adm_party_id,
        "unverified_party": decision.raw_name,
        "identity_score": round(item.rules_score, 6),
        "lexical_score": round(max(item.char_tfidf_score, item.word_tfidf_score, item.exact), 6),
        "decision_reason": decision.reason,
        "candidate_name": item.matched_name,
        "name_case": case,
        "matched_segment": item.mention_text,
        "connector_resolution": decision.connector_resolution or "",
        "status": "REVIEW_ONLY",
    } for item in selected]


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _held_out_check(prepared: Path, selected: list[PartyRecord], graph: VerifiedGraph,
                    strong: list[dict], review: list[dict]) -> dict | None:
    """Report only; held-out labels never select rows, candidates or outcomes."""
    path = prepared / "labels.jsonl"
    if not path.exists():
        return None
    selected_ids = {item.adm_party_id for item in selected}
    labels = {row["adm_party_id"]: row for row in read_jsonl(path)
              if row.get("adm_party_id") in selected_ids and row.get("scorable")
              and row.get("split") in {"TEST_KNOWN", "TEST_UNSEEN"}}
    expected: dict[str, str] = {}
    for adm_id, label in labels.items():
        party_id = str(label.get("expected_party_id") or "")
        expected[adm_id] = graph.root_id(party_id) if party_id in graph.nodes else party_id
    accepted = [row for row in strong if row["unverified_party_id"] in expected]
    correct = sum(row["verified_party_id"] == expected[row["unverified_party_id"]]
                  for row in accepted)
    review_hit_ids = {row["unverified_party_id"] for row in review
                      if row["unverified_party_id"] in expected
                      and row["verified_party_id"] == expected[row["unverified_party_id"]]}
    review_labeled = [row for row in review if row["unverified_party_id"] in expected]
    review_correct = sum(row["verified_party_id"] == expected[row["unverified_party_id"]]
                         for row in review_labeled)
    return {"scope": "selected held-out eligible rows only, not the full POC benchmark",
            "labeled_rows": len(labels), "strong_matches": len(accepted),
            "strong_correct": correct, "strong_wrong": len(accepted) - correct,
            "strong_precision": correct / len(accepted) if accepted else None,
            "strong_recall": correct / len(labels) if labels else None,
            "review_candidate_rows": len(review_labeled),
            "review_correct_candidate_rows": review_correct,
            "review_candidate_precision": review_correct / len(review_labeled) if review_labeled else None,
            "review_names_with_expected_candidate": len(review_hit_ids)}


def build(prepared: Path, config: dict, min_group_size: int = 0,
          max_groups: int = 0, focus_names: list[str] | None = None,
          added_names: list[str] | None = None, review_score_floor: float = 0.60,
          review_lexical_floor: float = 0.55,
          review_max_candidates: int = 2, review_within_top: float = 0.10,
          catalog_override: list[dict] | None = None,
          selected_record_ids: set[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    started = time.perf_counter()
    jobs = read_json(prepared / "jobs.json", []) or []
    if not jobs:
        raise ValueError("Prepared jobs.json is missing; run scripts/prepare.py first")
    account_id = str(jobs[0]["accountId"])
    if any(str(job["accountId"]) != account_id for job in jobs):
        raise ValueError("A suggestion run must contain one account")
    graph = _current_graph(prepared, config, account_id, added_names or [], catalog_override)
    eligible, excluded = _eligible_records(prepared)
    if selected_record_ids is not None:
        eligible = [item for item in eligible if item.adm_party_id in selected_record_ids]
    selected, groups = _scope(eligible, graph, min_group_size, max_groups, focus_names or [])
    if not selected:
        raise ValueError("No eligible unverified rows selected; relax the scope filter")
    available_cases = Counter(_record_case(item) for item in eligible)
    selected_cases = {item.adm_party_id: _record_case(item) for item in selected}
    selected_case_counts = Counter(selected_cases.values())
    decisions, proposals, scorer, stages = _rules_decisions(selected, graph, config, prepared, jobs)
    records_by_id = {item.adm_party_id: item for item in selected}
    source_rows = {item.adm_party_id: item.source_row for item in selected}
    strong: list[dict] = []
    review: list[dict] = []
    for decision in decisions:
        record = records_by_id[decision.adm_party_id]
        case = selected_cases[decision.adm_party_id]
        if decision.decision == "MATCH":
            strong.append({
                "verified_party_id": decision.verified_party_id,
                "verified_party": decision.verified_party_name,
                "unverified_party_id": decision.adm_party_id,
                "unverified_party": decision.raw_name,
                "source_row": source_rows[decision.adm_party_id],
                "rules_score": round(decision.confidence, 6),
                "match_method": decision.match_method or "",
                "decision_tier": decision.decision_tier or "",
                "name_case": case,
                "matched_segment": decision.matched_mention or "",
                "connector_resolution": decision.connector_resolution or "",
                "status": "STRONG_SUGGESTION",
            })
        else:
            for item in _review_candidates(
                    record, decision, proposals.get(decision.adm_party_id, []), config,
                    review_score_floor, review_lexical_floor, review_max_candidates,
                    review_within_top):
                item["source_row"] = source_rows[decision.adm_party_id]
                review.append(item)
    strong.sort(key=lambda row: (row["verified_party"].casefold(),
                                 row["unverified_party"].casefold(), row["unverified_party_id"]))
    review.sort(key=lambda row: (row["verified_party"].casefold(),
                                 row["unverified_party"].casefold(), row["unverified_party_id"]))
    strong_counts = Counter(row["verified_party_id"] for row in strong)
    review_counts = Counter(row["verified_party_id"] for row in review)
    for row in strong:
        row["verified_strong_count"] = strong_counts[row["verified_party_id"]]
    for row in review:
        row["verified_review_count"] = review_counts[row["verified_party_id"]]
    verified = [
        {"verified_party_id": party_id, "verified_party": node.party_name,
         "strong_count": strong_counts[party_id], "review_count": review_counts[party_id]}
        for party_id, node in graph.nodes.items()
        if strong_counts[party_id] or review_counts[party_id]
    ]
    verified.sort(key=lambda row: (-row["strong_count"], -row["review_count"],
                                   row["verified_party"].casefold()))
    summary = {
        "workflow": "poc_rules_suggestions",
        "scope": {"min_raw_anchor_group_size": min_group_size,
                  "max_anchor_groups": max_groups, "focus_verified": focus_names or [],
                  "selected_anchor_groups": groups,
                  "meaning": "Raw leading-name groups select input rows only; they are not known matches."},
        "counts": {"unverified_available": len(eligible), "unverified_scored": len(selected),
                   **{f"{case.lower()}_unverified_available": available_cases[case]
                      for case in ("PLAIN", "OBO", "VIA", "MIXED", "MALFORMED")},
                   **{f"{case.lower()}_unverified_scored": selected_case_counts[case]
                      for case in ("PLAIN", "OBO", "VIA", "MIXED", "MALFORMED")},
                   "strong_suggestions": len(strong), "review_candidate_rows": len(review),
                   "review_unverified_names": len({item["unverified_party_id"] for item in review}),
                   "no_suggestion_names": len(selected) - len(strong) - len({item["unverified_party_id"] for item in review}),
                   **dict(excluded)},
        "review_policy": {"minimum_identity_score": review_score_floor,
                          "minimum_lexical_score": review_lexical_floor,
                          "maximum_candidates_per_name": review_max_candidates,
                          "within_top_score": review_within_top,
                          "status": "Uncalibrated review filter, never an accepted match"},
        "rules": {"system_threshold": scorer.system_threshold,
                  "plain_threshold": scorer.plain_threshold,
                  "stages": stages},
        "verified_parties": verified,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "caveat": "Strong means accepted by this scoped rules-only POC, including its OBO/VIA connector policy; neither tier is a confirmed mapping or calibrated probability.",
    }
    held_out = _held_out_check(prepared, selected, graph, strong, review)
    if held_out is not None:
        summary["held_out_selected_only"] = held_out
    return strong, review, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--prepared-dir", type=Path,
                        help="Override paths.prepared_dir in config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-anchor-group-size", type=int, default=0,
                        help="Score only leading-name groups with at least N raw unverified rows; use 3 for more than two")
    parser.add_argument("--max-groups", type=int, default=0,
                        help="After the minimum, keep only the N largest leading-name groups; 0 means all")
    parser.add_argument("--focus-verified", action="append", default=[], metavar="NAME",
                        help="Focus input rows on this verified name's leading anchor, while all verified parties still compete")
    parser.add_argument("--add-verified", action="append", default=[], metavar="NAME",
                        help="Add a verified name in memory for this snapshot; no catalog or graph file is changed")
    parser.add_argument("--review-score-floor", type=float, default=0.60)
    parser.add_argument("--review-lexical-floor", type=float, default=0.55)
    parser.add_argument("--review-max-candidates", type=int, default=2)
    parser.add_argument("--review-within-top", type=float, default=0.10)
    parser.add_argument("--xlsx", action="store_true")
    parser.add_argument("--node", help="Node.js executable for XLSX")
    parser.add_argument("--artifact-modules", type=Path,
                        help="node_modules containing @oai/artifact-tool for XLSX")
    args = parser.parse_args()
    if args.min_anchor_group_size < 0 or args.max_groups < 0 or args.review_max_candidates < 1:
        parser.error("Group counts must be nonnegative; review candidate count must be positive")
    if not 0 <= args.review_score_floor <= 1 or not 0 <= args.review_lexical_floor <= 1 or not 0 <= args.review_within_top <= 1:
        parser.error("Review floors must be between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; existing results are preserved")
    bundled_node = (Path.home() / ".cache" / "codex-runtimes" /
                    "codex-primary-runtime" / "dependencies" / "node")
    bundled_exe = bundled_node / "bin" / ("node.exe" if os.name == "nt" else "node")
    node = args.node or (str(bundled_exe) if bundled_exe.exists() else shutil.which("node"))
    modules = args.artifact_modules or (bundled_node / "node_modules")
    if args.xlsx and (not node or not modules or not (modules / "@oai" / "artifact-tool").exists()):
        parser.error("XLSX requires Node.js and --artifact-modules containing @oai/artifact-tool")
    config = load_config(args.config)
    prepared = args.prepared_dir or Path(config.get("paths", {}).get("prepared_dir", "data/prepared"))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(config.get("execution", {}).get("native_threads", 10)))
    try:
        strong, review, summary = build(
            prepared, config, args.min_anchor_group_size, args.max_groups,
            args.focus_verified, args.add_verified, args.review_score_floor,
            args.review_lexical_floor, args.review_max_candidates, args.review_within_top)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "strong_suggestions.csv", strong,
               ("verified_party", "verified_strong_count", "unverified_party", "rules_score",
                "match_method", "decision_tier", "name_case", "matched_segment",
                "connector_resolution", "source_row", "verified_party_id",
                "unverified_party_id", "status"))
    _write_csv(args.output_dir / "review_candidates.csv", review,
               ("verified_party", "verified_review_count", "unverified_party", "identity_score",
                "lexical_score", "decision_reason", "candidate_name", "name_case",
                "matched_segment", "connector_resolution", "source_row",
                "verified_party_id", "unverified_party_id", "status"))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.xlsx:
        from .current import _export_xlsx
        _export_xlsx(args.output_dir, node, modules, strong, review, summary)
    print(f"Scored {summary['counts']['unverified_scored']} of {summary['counts']['unverified_available']} eligible unverified names (plain, OBO and VIA)")
    print(f"Strong suggestions: {len(strong)}; review candidate rows: {len(review)}")
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
