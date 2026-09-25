from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import xlsxwriter

from .domain import normalize_name, read_json, read_jsonl, write_json


DETAIL_COLUMNS = ["Unverified Party", "Verified Party - Matched", "Verified Party - Label", "Result", "Rules-only prediction"]


def build_reports(
    prepared_dir: str | Path, output_dir: str | Path, run_stats: dict[str, Any],
    rules_demo_only: bool = False,
) -> dict[str, Any]:
    prepared, output = Path(prepared_dir), Path(output_dir)
    manifest = read_json(prepared / "manifest.json", {}) or {}
    labels = {row["adm_party_id"]: row for row in read_jsonl(prepared / "labels.jsonl")}
    decisions = {row["adm_party_id"]: row for row in read_jsonl(output / "decisions.jsonl")}
    rules_path = output / "rules_decisions.jsonl"
    rules_decisions = {row["adm_party_id"]: row for row in read_jsonl(rules_path)} if rules_path.exists() else {}
    prior_rules_path = output / "rules_baseline_decisions.jsonl"
    prior_rules_decisions = {row["adm_party_id"]: row for row in read_jsonl(prior_rules_path)} if prior_rules_path.exists() else {}
    source_rows = list(read_jsonl(prepared / "source_rows.jsonl"))
    graph_path = run_stats.get("graph_path")
    graph_payload = (read_json(graph_path, {}) or {}) if graph_path else {}
    graph_nodes = {str(node["party_id"]): node for node in graph_payload.get("nodes", [])}

    def global_parent(party_id: str | None) -> tuple[str | None, str | None]:
        if not party_id:
            return None, None
        current, seen = party_id, set()
        while current in graph_nodes and graph_nodes[current].get("parent_id"):
            if current in seen:
                raise ValueError(f"Cycle in evaluation graph at {current}")
            seen.add(current)
            current = str(graph_nodes[current]["parent_id"])
        return current, graph_nodes.get(current, {}).get("party_name")

    detail_rows = []
    for source_row in source_rows:
        adm_id = source_row["adm_party_id"]
        if adm_id not in decisions:
            continue
        label = labels[adm_id]
        decision = decisions[adm_id]
        rules_decision = rules_decisions.get(adm_id, {})
        prior_rules_decision = prior_rules_decisions.get(adm_id, {})
        rules_match = rules_decision.get("decision") == "MATCH"
        workbook_canonical_id = label.get("expected_party_id")
        expected, expected_parent_name = global_parent(workbook_canonical_id)
        candidate_id = decision.get("verified_party_id")
        candidate_name = decision.get("verified_party_name")
        is_match = decision.get("decision") == "MATCH"
        predicted = candidate_id if is_match else None
        correct = (predicted == expected) if label.get("scorable") and is_match else (False if label.get("scorable") else None)
        expected_retrieved = expected in set(decision.get("retrieved_root_ids") or []) if expected else None
        retrieved_roots = list(decision.get("retrieved_root_ids") or [])
        expected_rank = (retrieved_roots.index(expected) + 1) if expected and expected in retrieved_roots else None
        split = str(label.get("split") or "")
        held_out = bool(label.get("scorable")) and split in {"TEST_KNOWN", "TEST_UNSEEN"}
        raw_name = str(source_row["source"].get(manifest.get("raw_name_column")) or "")
        has_obo = bool(re.search(r"\bOBO\b", raw_name, flags=re.IGNORECASE))
        has_via = bool(re.search(r"\bVIA\b", raw_name, flags=re.IGNORECASE))
        prediction = {
            "ADM Party ID": adm_id,
            "Source Row": source_row["source_row"],
            "Dataset Split": split,
            "Category": label.get("category"),
            "Evaluation Status": (
                "UNKNOWN" if split == "UNKNOWN" else
                "LABELED_HELD_OUT" if held_out else
                "LABELED_DEVELOPMENT" if label.get("scorable") else "INELIGIBLE"
            ),
            "Held-Out": held_out,
            "Scorable": bool(label.get("scorable")),
            "Exclusion Reason": label.get("exclusion_reason"),
            "Workbook Canonical ID": workbook_canonical_id,
            "Expected Verified ID": expected,
            "Expected Canonical Name": label.get("expected_canonical_name"),
            "Expected Global Parent Name": expected_parent_name,
            "Decision": decision.get("decision"),
            "Correct": correct,
            "Confidence": decision.get("confidence"),
            "Reason": decision.get("reason"),
            "Event Emitted": bool(decision.get("emitted")),
            "Predicted Verified ID": predicted,
            "Predicted Verified Name": candidate_name if is_match else None,
            "Rules-only Verified ID": rules_decision.get("verified_party_id") if rules_match else None,
            "Rules-only prediction": rules_decision.get("verified_party_name") if rules_match else ("NO_MATCH" if rules_decision else None),
            "Rules-only Decision Tier": rules_decision.get("decision_tier"),
            "Prior Rules Verified ID": prior_rules_decision.get("verified_party_id") if prior_rules_decision.get("decision") == "MATCH" else None,
            "Prior Rules Decision Tier": prior_rules_decision.get("decision_tier"),
            "Top Candidate Verified ID": candidate_id,
            "Top Candidate Verified Name": candidate_name,
            "Matched Member ID": decision.get("matched_member_id"),
            "Matched Member Name": decision.get("matched_member_name"),
            "Matched Candidate Name": decision.get("matched_candidate_name"),
            "Candidate Type": decision.get("matched_candidate_type"),
            "Candidate Expansion Confidence": decision.get("candidate_expansion_confidence"),
            "Matched Mention": decision.get("matched_mention"),
            "Connector": decision.get("connector"),
            "Match Method": decision.get("match_method"),
            "Decision Tier": decision.get("decision_tier"),
            "Retrieval Sources": _join(decision.get("retrieval_sources")),
            "Identity Score": decision.get("identity_score"),
            "Char TF-IDF Score": decision.get("char_tfidf_score"),
            "Word TF-IDF Score": decision.get("word_tfidf_score"),
            "RRF Score": decision.get("rrf_score"),
            "Char Similarity": decision.get("char_similarity"),
            "Jaro-Winkler": decision.get("jaro_winkler"),
            "Levenshtein": decision.get("levenshtein"),
            "Token Jaccard": decision.get("token_jaccard"),
            "Raw Coverage": decision.get("raw_coverage"),
            "Candidate Coverage": decision.get("candidate_coverage"),
            "Distinctive Token Conflict": decision.get("distinctive_token_conflict"),
            "Digit Conflict": decision.get("digit_conflict"),
            "Cross-Encoder Score": decision.get("cross_encoder_score"),
            "Runner-Up Verified ID": decision.get("runner_up_party_id"),
            "Runner-Up Score": decision.get("runner_up_score"),
            "Margin": decision.get("margin"),
            "Graph Path": _join(decision.get("graph_path")),
            "Retrieved Root IDs": _join(retrieved_roots),
            "Retrieved Root Count": len(retrieved_roots),
            "Expected Target Retrieved": expected_retrieved,
            "Expected Target Rank": expected_rank,
            "Expected Segment Seen": any(
                item.get("root_id") == expected for item in decision.get("mention_results") or []
            ) if expected else False,
            "Has OBO": has_obo,
            "Has VIA": has_via,
            "Parse Warning": decision.get("parse_warning"),
            "Connector Resolution": decision.get("connector_resolution"),
            "Selected Mention Position": decision.get("selected_mention_position"),
            "Provisional Verified ID": decision.get("provisional_party_id"),
            "Provisional Verified Name": decision.get("provisional_party_name"),
            "Mention Results": " | ".join(
                f"{item['position']}: {item['text']} -> {item.get('root_name') or 'NO_MATCH'} "
                f"({item['reason']}, {item['confidence']:.3f})"
                for item in decision.get("mention_results") or []
            ),
            "Strategy": run_stats.get("strategy"),
            "Model Version": run_stats.get("model_version"),
            "Expansion Version": run_stats.get("expansion_version"),
            "Graph Schema Version": run_stats.get("graph_schema_version"),
            "Run ID": run_stats.get("run_id"),
            "System Threshold": run_stats.get("system_threshold"),
            "Effective Cutoff": run_stats.get("effective_cutoff"),
        }
        prediction["Error Bucket"] = _error_bucket(prediction)
        prediction["Case Types"] = _join(_case_types(raw_name, prediction))
        detail_rows.append({"source": source_row["source"], "prediction": prediction})
    if rules_demo_only and not rules_decisions:
        raise FileNotFoundError(f"Rules decisions not found: {rules_path}")
    if rules_demo_only:
        rules_rows = _rules_demo_rows(detail_rows, rules_decisions, manifest)
        rules_metrics = _metrics(rules_rows, run_stats)
        _write_workbook(output / "predictions_rules_only.xlsx", rules_rows, manifest,
                        rules_metrics, run_stats, view="rules")
        return rules_metrics
    _write_rules_changes(output / "rules_changes.csv", detail_rows, manifest,
                         prior_rules_decisions, rules_decisions)
    metrics = _metrics(detail_rows, run_stats)
    write_json(output / "run_metrics.json", metrics)
    write_json(output / "diagnostics.json", {
        "headline": metrics["overall"],
        "cohorts": metrics["cohorts"],
        "funnel": metrics["funnel"],
        "error_buckets": metrics["error_buckets"],
        "candidate_retrieval": metrics["candidate_retrieval"],
        "unknown_predictions": metrics["unknown_predictions"],
    })
    _write_analysis(output / "analysis.md", metrics, run_stats)
    _write_workbook(output / "predictions.xlsx", detail_rows, manifest, metrics, run_stats)
    return metrics


def _rules_demo_rows(
    detail_rows: list[dict[str, Any]], rules_decisions: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project rules decisions into the established report schema; do not mutate ML rows."""
    projected = []
    for row in detail_rows:
        ml = row["prediction"]
        rule = rules_decisions.get(ml["ADM Party ID"])
        if rule is None:
            raise ValueError(f"Missing rules decision for {ml['ADM Party ID']}")
        prediction = dict(ml)
        is_match = rule.get("decision") == "MATCH"
        target = rule.get("verified_party_id") if is_match else None
        expected = prediction.get("Expected Verified ID")
        retrieved = list(rule.get("retrieved_root_ids") or [])
        prediction.update({
            "Decision": rule.get("decision"),
            "Correct": (target == expected) if prediction["Scorable"] and is_match
                       else (False if prediction["Scorable"] else None),
            "Confidence": rule.get("confidence"),
            "Reason": rule.get("reason"),
            "Event Emitted": False,
            "Predicted Verified ID": target,
            "Predicted Verified Name": rule.get("verified_party_name") if is_match else None,
            "Expected Target Retrieved": expected in retrieved if expected else None,
            "Expected Target Rank": retrieved.index(expected) + 1 if expected in retrieved else None,
            "Expected Segment Seen": any(item.get("root_id") == expected
                                         for item in rule.get("mention_results") or []) if expected else False,
            "Connector Resolution": rule.get("connector_resolution"),
            "Selected Mention Position": rule.get("selected_mention_position"),
            "Rules-only Verified ID": target,
            "Rules-only prediction": rule.get("verified_party_name") if is_match else "NO_MATCH",
            "Rules-only Decision Tier": rule.get("decision_tier"),
        })
        prediction["Error Bucket"] = _error_bucket(prediction)
        raw_name = str(row["source"].get(manifest.get("raw_name_column")) or "")
        prediction["Case Types"] = _join(_case_types(raw_name, prediction))
        projected.append({"source": row["source"], "prediction": prediction})
    return projected


def _write_rules_changes(
    path: Path, detail_rows: list[dict[str, Any]], manifest: dict[str, Any],
    prior: dict[str, dict[str, Any]], enhanced: dict[str, dict[str, Any]],
) -> None:
    """Small, label-aware audit only; matching itself never consumes labels."""
    columns = ["Split", "Unverified Party", "Verified Party - Label", "Previous rules",
               "Enhanced rules", "Enhanced result", "Enhanced decision tier"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in detail_rows:
            prediction = row["prediction"]
            adm_id = prediction["ADM Party ID"]
            old, new = prior.get(adm_id), enhanced.get(adm_id)
            if not old or not new:
                continue
            old_target = old.get("verified_party_id") if old.get("decision") == "MATCH" else None
            new_target = new.get("verified_party_id") if new.get("decision") == "MATCH" else None
            if old_target == new_target:
                continue
            expected = prediction.get("Expected Verified ID")
            result = ("UNSCORED" if not prediction.get("Scorable") else
                      "NO_MATCH" if new_target is None else
                      "CORRECT" if new_target == expected else "WRONG")
            writer.writerow([
                prediction["Dataset Split"],
                row["source"].get(manifest.get("raw_name_column")) or "",
                prediction.get("Expected Canonical Name") or "",
                old.get("verified_party_name") if old_target else "NO_MATCH",
                new.get("verified_party_name") if new_target else "NO_MATCH",
                result, new.get("decision_tier") or "",
            ])


def _metrics(detail_rows: list[dict[str, Any]], run_stats: dict[str, Any]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scorable = [row for row in rows if row["prediction"]["Scorable"]]
        matches = [row for row in rows if row["prediction"]["Decision"] == "MATCH"]
        scored_matches = [row for row in matches if row["prediction"]["Scorable"]]
        correct = [row for row in scored_matches if row["prediction"]["Correct"]]
        retrieved = [row for row in scorable if row["prediction"]["Expected Target Retrieved"]]
        interval = _wilson_interval(len(correct), len(scored_matches)) if scored_matches else (None, None)
        precision = len(correct) / len(scored_matches) if scored_matches else None
        recall = len(correct) / len(scorable) if scorable else None
        return {
            "rows": len(rows),
            "scorable": len(scorable),
            "matches": len(matches),
            "correct_matches": len(correct),
            "precision": precision,
            "precision_ci_95_low": interval[0],
            "precision_ci_95_high": interval[1],
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
            # Backward-compatible alias for any existing consumers of metrics.json.
            "correct_match_recall": len(correct) / len(scorable) if scorable else None,
            "coverage": len(scored_matches) / len(scorable) if scorable else None,
            "retrieval_recall": len(retrieved) / len(scorable) if scorable else None,
        }
    held_out = [row for row in detail_rows if row["prediction"]["Held-Out"]]
    all_labeled = [row for row in detail_rows if row["prediction"]["Scorable"]]
    all_labeled_excluding_error = [
        row for row in all_labeled
        if str(row["prediction"].get("Category") or "").strip().casefold() != "error"
    ]
    split_names = sorted({row["prediction"]["Dataset Split"] for row in detail_rows})
    by_case: dict[str, Any] = {}
    case_names = sorted({
        case
        for row in detail_rows
        for case in str(row["prediction"].get("Case Types") or "").split(" | ")
        if case
    })
    for case in case_names:
        by_case[case] = summarize([
            row for row in held_out
            if case in str(row["prediction"].get("Case Types") or "").split(" | ")
        ])
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in held_out:
        category_rows[str(row["prediction"].get("Category") or "(blank)")].append(row)
    by_category = {category: summarize(category_rows[category]) for category in sorted(category_rows)}
    cohorts = {
        "ALL_HELD_OUT": summarize(held_out),
        "NO_OBO": summarize([row for row in held_out if not row["prediction"]["Has OBO"]]),
        "NO_VIA": summarize([row for row in held_out if not row["prediction"]["Has VIA"]]),
        "PLAIN": summarize([
            row for row in held_out
            if not row["prediction"]["Has OBO"] and not row["prediction"]["Has VIA"]
        ]),
        "OBO_ONLY": summarize([
            row for row in held_out
            if row["prediction"]["Has OBO"] and not row["prediction"]["Has VIA"]
        ]),
        "VIA_ONLY": summarize([
            row for row in held_out
            if row["prediction"]["Has VIA"] and not row["prediction"]["Has OBO"]
        ]),
        "OBO_AND_VIA": summarize([
            row for row in held_out
            if row["prediction"]["Has OBO"] and row["prediction"]["Has VIA"]
        ]),
        "CONNECTOR_ONE_VALID": summarize([row for row in held_out if row["prediction"]["Connector Resolution"] == "ONLY_MATCH"]),
        "CONNECTOR_SAME_ROOT": summarize([row for row in held_out if row["prediction"]["Connector Resolution"] == "SAME_ROOT"]),
        "CONNECTOR_CONFLICT": summarize([row for row in held_out if str(row["prediction"]["Connector Resolution"] or "").startswith(("OBO_", "VIA_"))]),
        "CONNECTOR_ABSTAINED": summarize([
            row for row in held_out
            if (row["prediction"]["Has OBO"] or row["prediction"]["Has VIA"])
            and row["prediction"]["Decision"] == "NO_MATCH"
        ]),
        "PREFERRED_NEAR_CUTOFF": summarize([
            row for row in held_out if row["prediction"]["Reason"] == "PREFERRED_SEGMENT_NEAR_CUTOFF"
        ]),
    }
    bins = []
    for lower in np_arange(0.0, 1.0, 0.1):
        upper = round(lower + 0.1, 10)
        bucket = [
            row for row in held_out
            if row["prediction"]["Scorable"] and row["prediction"]["Decision"] == "MATCH"
            and float(row["prediction"]["Confidence"] or 0.0) >= lower
            and (float(row["prediction"]["Confidence"] or 0.0) < upper or upper >= 1.0)
        ]
        if bucket:
            bins.append({
                "bucket": f"{lower:.1f}-{upper:.1f}",
                "matches": len(bucket),
                "average_confidence": sum(float(row["prediction"]["Confidence"]) for row in bucket) / len(bucket),
                "accuracy": sum(bool(row["prediction"]["Correct"]) for row in bucket) / len(bucket),
            })
    unknown = [row for row in detail_rows if row["prediction"]["Evaluation Status"] == "UNKNOWN"]
    unknown_matches = [row for row in unknown if row["prediction"]["Decision"] == "MATCH"]
    funnel = {
        "held_out_labeled": len(held_out),
        "expected_retrieved": sum(bool(row["prediction"]["Expected Target Retrieved"]) for row in held_out),
        "expected_rank_1": sum(row["prediction"]["Expected Target Rank"] == 1 for row in held_out),
        "accepted_matches": sum(row["prediction"]["Decision"] == "MATCH" for row in held_out),
        "correct_accepted": sum(bool(row["prediction"]["Correct"]) for row in held_out),
        "wrong_accepted": sum(
            row["prediction"]["Decision"] == "MATCH" and not row["prediction"]["Correct"]
            for row in held_out
        ),
        "retrieval_misses": sum(row["prediction"]["Error Bucket"] == "RETRIEVAL_MISS" for row in held_out),
        "rejected_after_retrieval": sum(
            row["prediction"]["Error Bucket"] == "REJECTED_AFTER_RETRIEVAL" for row in held_out
        ),
        "ranking_errors": sum(row["prediction"]["Error Bucket"] == "RANKING_ERROR" for row in held_out),
    }
    canonical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in held_out:
        canonical_groups[str(row["prediction"].get("Expected Canonical Name") or "UNKNOWN")].append(row)
    entity_recalls = [
        sum(bool(row["prediction"]["Correct"]) for row in rows) / len(rows)
        for rows in canonical_groups.values() if rows
    ]
    rules_matches = [row for row in held_out if row["prediction"].get("Rules-only Verified ID")]
    rules_correct = sum(
        row["prediction"]["Rules-only Verified ID"] == row["prediction"]["Expected Verified ID"]
        for row in rules_matches
    )
    rules_precision = rules_correct / len(rules_matches) if rules_matches else None
    rules_recall = rules_correct / len(held_out) if held_out else None
    pre_core_matches = [row for row in rules_matches
                        if row["prediction"].get("Rules-only Decision Tier") != "RULES_SAFE_VIEW_CORE_ANCHOR"]
    pre_core_correct = sum(
        row["prediction"]["Rules-only Verified ID"] == row["prediction"]["Expected Verified ID"]
        for row in pre_core_matches
    )
    pre_core_precision = pre_core_correct / len(pre_core_matches) if pre_core_matches else None
    pre_core_recall = pre_core_correct / len(held_out) if held_out else None
    prior_rules_matches = [row for row in held_out if row["prediction"].get("Prior Rules Verified ID")]
    prior_rules_correct = sum(
        row["prediction"]["Prior Rules Verified ID"] == row["prediction"]["Expected Verified ID"]
        for row in prior_rules_matches
    )
    prior_rules_precision = prior_rules_correct / len(prior_rules_matches) if prior_rules_matches else None
    prior_rules_recall = prior_rules_correct / len(held_out) if held_out else None
    baseline_matches = [row for row in prior_rules_matches
                        if row["prediction"].get("Prior Rules Decision Tier") != "RULES_UNIQUE_CONTAINMENT"]
    baseline_correct = sum(
        row["prediction"]["Prior Rules Verified ID"] == row["prediction"]["Expected Verified ID"]
        for row in baseline_matches
    )
    baseline_precision = baseline_correct / len(baseline_matches) if baseline_matches else None
    baseline_recall = baseline_correct / len(held_out) if held_out else None

    def containment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        selected = [row["prediction"] for row in rows
                    if row["prediction"].get("Rules-only Verified ID")
                    and row["prediction"].get("Rules-only Decision Tier") == "RULES_UNIQUE_CONTAINMENT"]
        correct = sum(row["Rules-only Verified ID"] == row["Expected Verified ID"] for row in selected)
        return {"matches": len(selected), "correct": correct,
                "precision": correct / len(selected) if selected else None}
    return {
        "overall": summarize(held_out),
        "rules_only": {
            "matches": len(rules_matches),
            "correct_matches": rules_correct,
            "precision": rules_precision,
            "recall": rules_recall,
            "f1": 2 * rules_precision * rules_recall / (rules_precision + rules_recall)
            if rules_precision is not None and rules_recall is not None and rules_precision + rules_recall else None,
            "disagreements": sum(
                row["prediction"].get("Predicted Verified ID") != row["prediction"].get("Rules-only Verified ID")
                for row in held_out
            ),
            "before_core_anchor": {
                "matches": len(pre_core_matches), "correct_matches": pre_core_correct,
                "precision": pre_core_precision, "recall": pre_core_recall,
                "f1": 2 * pre_core_precision * pre_core_recall / (pre_core_precision + pre_core_recall)
                if pre_core_precision is not None and pre_core_recall is not None
                and pre_core_precision + pre_core_recall else None,
            },
            "prior_rules": {
                "matches": len(prior_rules_matches), "correct_matches": prior_rules_correct,
                "precision": prior_rules_precision, "recall": prior_rules_recall,
                "f1": 2 * prior_rules_precision * prior_rules_recall / (prior_rules_precision + prior_rules_recall)
                if prior_rules_precision is not None and prior_rules_recall is not None
                and prior_rules_precision + prior_rules_recall else None,
            },
            "containment": {
                "calibration": containment_summary([
                    row for row in detail_rows if row["prediction"]["Dataset Split"] == "CALIBRATION"
                    and row["prediction"]["Scorable"]
                ]),
                "held_out": containment_summary(held_out),
            },
            "baseline_without_containment": {
                "matches": len(baseline_matches), "correct_matches": baseline_correct,
                "precision": baseline_precision, "recall": baseline_recall,
                "f1": 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall)
                if baseline_precision is not None and baseline_recall is not None
                and baseline_precision + baseline_recall else None,
            },
        },
        "all_labeled": summarize(all_labeled),
        "all_labeled_excluding_error": summarize(all_labeled_excluding_error),
        "cohorts": cohorts,
        "by_split": {
            split: summarize([row for row in detail_rows if row["prediction"]["Dataset Split"] == split])
            for split in split_names
        },
        "by_case": by_case,
        "by_category": by_category,
        "funnel": funnel,
        "error_buckets": dict(Counter(row["prediction"]["Error Bucket"] for row in held_out)),
        "macro_entity_recall": sum(entity_recalls) / len(entity_recalls) if entity_recalls else None,
        "evaluated_entities": len(entity_recalls),
        "unknown_predictions": {
            "rows": len(unknown),
            "matches": len(unknown_matches),
            "no_matches": len(unknown) - len(unknown_matches),
            "prediction_rate": len(unknown_matches) / len(unknown) if unknown else None,
            "average_match_confidence": (
                sum(float(row["prediction"]["Confidence"] or 0.0) for row in unknown_matches) / len(unknown_matches)
                if unknown_matches else None
            ),
        },
        "candidate_retrieval": {
            "raw_hits": run_stats.get("raw_retrieval_hits"),
            "scored_proposals": run_stats.get("scored_proposals"),
            "mentions_with_candidates": run_stats.get("mentions_with_candidates"),
            "root_cap_hits": run_stats.get("root_cap_hits"),
            "variant_cap_hits": run_stats.get("variant_cap_hits"),
            "candidate_roots_p50": run_stats.get("candidate_roots_p50"),
            "candidate_roots_p95": run_stats.get("candidate_roots_p95"),
            "candidate_roots_max": run_stats.get("candidate_roots_max"),
        },
        "confidence_bins": bins,
        "no_match_reasons": dict(Counter(
            row["prediction"]["Reason"] for row in detail_rows
            if row["prediction"]["Held-Out"] and row["prediction"]["Decision"] == "NO_MATCH"
        )),
        "runtime": run_stats,
    }


def _write_workbook(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    run_stats: dict[str, Any],
    view: str = "ml",
) -> None:
    if view not in {"ml", "rules"}:
        raise ValueError(f"Unknown workbook view: {view}")
    is_rules = view == "rules"
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False})
    workbook.set_properties({"title": "Rules-only party matching predictions" if is_rules else "Party matching predictions",
                             "subject": "POC evaluation output"})
    title = workbook.add_format({"font_name": "Arial", "font_size": 14, "bold": True, "font_color": "#1F2937"})
    section = workbook.add_format({"font_name": "Arial", "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"})
    header = workbook.add_format({
        "font_name": "Arial", "font_size": 10, "bold": True, "font_color": "#FFFFFF",
        "bg_color": "#243B53", "align": "center", "valign": "vcenter", "border": 0, "text_wrap": True,
    })
    text = workbook.add_format({"font_name": "Arial", "font_size": 10, "font_color": "#1F2937"})
    integer = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "#,##0"})
    decimal = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000"})
    percent = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.0%"})

    grouped = workbook.add_worksheet("Summary")
    grouped.hide_gridlines(2)
    grouped.set_tab_color("#1F4E78")
    grouped.freeze_panes(1, 1)
    grouped.write_row(0, 0, [
        "Verified Party", "Number in dataset", "Number of unverified parties matched",
        "Number of correct predictions", "Unverified Parties",
        "Unverified Parties - Ground truth",
    ], header)
    raw_name_column = manifest.get("raw_name_column")
    by_party: dict[str, dict[str, Any]] = {}
    for row in rows:
        prediction = row["prediction"]
        raw = str(row["source"].get(raw_name_column) or "(blank raw name)")
        expected_id = prediction.get("Expected Verified ID")
        if expected_id and str(prediction.get("Expected Canonical Name") or "").strip().upper() != "UNKNOWN":
            group = by_party.setdefault(expected_id, {"name": prediction.get("Expected Global Parent Name") or prediction.get("Expected Canonical Name"), "truth": [], "matched": [], "correct": 0})
            group["truth"].append(raw)
        if prediction["Decision"] == "MATCH":
            party_id = prediction.get("Predicted Verified ID")
            if party_id:
                group = by_party.setdefault(party_id, {"name": prediction.get("Predicted Verified Name"), "truth": [], "matched": [], "correct": 0})
                group["matched"].append(raw)
                group["correct"] += int(bool(prediction.get("Correct")))
    row_number = 1
    for group in sorted(by_party.values(), key=lambda item: (-len(item["truth"]), -len(item["matched"]), str(item["name"]).casefold())):
        matched_chunks = _text_chunks(group["matched"])
        truth_chunks = _text_chunks(group["truth"])
        for part in range(max(len(matched_chunks), len(truth_chunks), 1)):
            if part == 0:
                grouped.write_string(row_number, 0, str(group["name"]), text)
                grouped.write_number(row_number, 1, len(group["truth"]), integer)
                grouped.write_number(row_number, 2, len(group["matched"]), integer)
                grouped.write_number(row_number, 3, group["correct"], integer)
            if part < len(matched_chunks):
                grouped.write_string(row_number, 4, matched_chunks[part], text)
            if part < len(truth_chunks):
                grouped.write_string(row_number, 5, truth_chunks[part], text)
            row_number += 1
    grouped.set_row(0, 38)
    grouped.set_column("A:A", 48)
    grouped.set_column("B:B", 16)
    grouped.set_column("C:D", 24)
    grouped.set_column("E:F", 110)
    if row_number > 1:
        grouped.autofilter(0, 0, row_number - 1, 5)

    summary = workbook.add_worksheet("Stats")
    summary.hide_gridlines(2)
    summary.set_tab_color("#1F4E78")
    summary.write("A2", "Rules-only party matching evaluation" if is_rules else "Party matching evaluation", title)
    _section_band(summary, 3, 0, 1, "Run summary", section)
    _section_band(summary, 3, 3, 8, "Rules-only evaluation" if is_rules else "Held-out ML versus rules-only", section)
    if not is_rules:
        _section_band(summary, 3, 11, 12, "Matching stage profile", section)
    summary.write_row("A5", ["Metric", "Value"], header)
    summary.write_row(4, 3, ["Method", "Matches", "Correct", "Precision", "Recall", "F1"], header)
    if not is_rules:
        summary.write_row(4, 11, ["Stage", "Seconds / MB"], header)
    overall = metrics["overall"]
    method_rows = (
        (
            (5, "Held-out enhanced rules", overall),
            (6, "All labeled excl. ERROR", metrics["all_labeled_excluding_error"]),
        ) if is_rules else (
            (5, "ML", overall),
            (6, "Rules only - enhanced", metrics["rules_only"]),
            (7, "Rules only - before core anchor", metrics["rules_only"]["before_core_anchor"]),
            (8, "Rules only - previous", metrics["rules_only"]["prior_rules"]),
            (9, "Rules only - no containment", metrics["rules_only"]["baseline_without_containment"]),
        )
    )
    for row_number, name, values in method_rows:
        summary.write_string(row_number, 3, name, text)
        summary.write_number(row_number, 4, values["matches"], integer)
        summary.write_number(row_number, 5, values["correct_matches"], integer)
        for column, key in ((6, "precision"), (7, "recall"), (8, "f1")):
            if values[key] is not None:
                summary.write_number(row_number, column, values[key], percent)
    cleaning = manifest.get("cleaning_counts", {})
    summary_values = [
        ("Original workbook rows", manifest.get("rows"), integer),
        ("Cleaned unverified rows", manifest.get("cleaned_unverified_rows", run_stats.get("records")), integer),
        ("Exact duplicate rows excluded", cleaning.get("EXACT_DUPLICATE", 0), integer),
        ("Verified self-rows excluded", cleaning.get("VERIFIED_SELF_ROW", 0), integer),
        ("Conflicting-label rows excluded", cleaning.get("CONFLICTING_LABELS", 0), integer),
        ("Held-out labeled rows", overall["scorable"], integer),
        ("Held-out matches", overall["matches"], integer),
        ("Held-out correct matches", overall["correct_matches"], integer),
        ("Events written", run_stats.get("events"), integer),
        ("Held-out precision", overall["precision"], percent),
        ("Precision 95% CI low", overall["precision_ci_95_low"], percent),
        ("Precision 95% CI high", overall["precision_ci_95_high"], percent),
        ("Held-out recall", overall["recall"], percent),
        ("Held-out F1", overall["f1"], percent),
        ("Held-out macro entity recall", metrics.get("macro_entity_recall"), percent),
        ("Held-out coverage", overall["coverage"], percent),
        ("Held-out retrieval recall", overall["retrieval_recall"], percent),
        ("Unknown rows (not scored)", metrics["unknown_predictions"]["rows"], integer),
        ("Unknown predicted matches", metrics["unknown_predictions"]["matches"], integer),
        ("Matching runtime (seconds)", run_stats.get("total_seconds"), decimal),
        ("System threshold", run_stats.get("system_threshold"), decimal),
        ("Effective cutoff", run_stats.get("effective_cutoff"), decimal),
        ("Rules-only connector cutoff", run_stats.get("rules_system_threshold"), decimal),
        ("Rules-only plain cutoff", run_stats.get("rules_plain_threshold", run_stats.get("rules_system_threshold")), decimal),
        ("Enhanced rules enabled", "yes" if run_stats.get("rules_enhanced_enabled") else "no", text),
        ("Rules multiword prefix enabled", "yes" if run_stats.get("rules_multiword_prefix_enabled") else "no", text),
        ("Enhanced rules short-name matches", (run_stats.get("rules_enhancement") or {}).get("short_name_matches"), integer),
        ("Enhanced rules multiword prefix matches", (run_stats.get("rules_enhancement") or {}).get("multiword_short_name_matches"), integer),
        ("Rules connector multiword prefix matches", run_stats.get("rules_connector_multiword_matches"), integer),
        ("Rules connector ambiguous-prefix abstentions", run_stats.get("rules_connector_ambiguous_prefix_abstentions"), integer),
        ("Enhanced rules safe-view matches", (run_stats.get("rules_enhancement") or {}).get("safe_view_matches"), integer),
        ("Enhanced rules core-anchor matches", (run_stats.get("rules_enhancement") or {}).get("core_anchor_matches"), integer),
        ("Rules containment enabled", "yes" if run_stats.get("rules_containment_enabled") else "no", text),
        ("Rules-only containment floor", run_stats.get("rules_containment_min_confidence"), decimal),
        ("Rules containment calibration matches", metrics["rules_only"]["containment"]["calibration"]["matches"], integer),
        ("Rules containment calibration correct", metrics["rules_only"]["containment"]["calibration"]["correct"], integer),
        ("Rules containment calibration precision", metrics["rules_only"]["containment"]["calibration"]["precision"], percent),
        ("Rules containment held-out matches", metrics["rules_only"]["containment"]["held_out"]["matches"], integer),
        ("Rules containment held-out correct", metrics["rules_only"]["containment"]["held_out"]["correct"], integer),
        ("Rules containment held-out precision", metrics["rules_only"]["containment"]["held_out"]["precision"], percent),
        ("Held-out ML/rules disagreements", metrics["rules_only"]["disagreements"], integer),
        ("Held-out preferred-part abstentions", metrics["cohorts"]["PREFERRED_NEAR_CUTOFF"]["rows"], integer),
    ]
    if is_rules:
        # The demo workbook is self-contained: all counts and rates come from
        # rules decisions, with no ML events, cutoffs, or comparison rows.
        summary_values = [
            ("Original workbook rows", manifest.get("rows"), integer),
            ("Cleaned unverified rows", manifest.get("cleaned_unverified_rows", run_stats.get("records")), integer),
            ("Exact duplicate rows excluded", cleaning.get("EXACT_DUPLICATE", 0), integer),
            ("Verified self-rows excluded", cleaning.get("VERIFIED_SELF_ROW", 0), integer),
            ("Conflicting-label rows excluded", cleaning.get("CONFLICTING_LABELS", 0), integer),
            ("Held-out labeled rows", overall["scorable"], integer),
            ("Held-out rules matches", overall["matches"], integer),
            ("Held-out correct matches", overall["correct_matches"], integer),
            ("Held-out precision", overall["precision"], percent),
            ("Precision 95% CI low", overall["precision_ci_95_low"], percent),
            ("Precision 95% CI high", overall["precision_ci_95_high"], percent),
            ("Held-out recall", overall["recall"], percent),
            ("Held-out F1", overall["f1"], percent),
            ("All labeled rows (excl. ERROR)", metrics["all_labeled_excluding_error"]["scorable"], integer),
            ("All labeled correct (excl. ERROR)", metrics["all_labeled_excluding_error"]["correct_matches"], integer),
            ("All labeled recall (excl. ERROR)", metrics["all_labeled_excluding_error"]["recall"], percent),
            ("Held-out coverage", overall["coverage"], percent),
            ("Held-out retrieval recall", overall["retrieval_recall"], percent),
            ("Unknown rows (not scored)", metrics["unknown_predictions"]["rows"], integer),
            ("Unknown rules matches", metrics["unknown_predictions"]["matches"], integer),
            ("Rules-only connector cutoff", run_stats.get("rules_system_threshold"), decimal),
            ("Rules-only plain cutoff", run_stats.get("rules_plain_threshold"), decimal),
            ("Rules enhancement enabled", "yes" if run_stats.get("rules_enhanced_enabled") else "no", text),
            ("Rules multiword prefix enabled", "yes" if run_stats.get("rules_multiword_prefix_enabled") else "no", text),
            ("Rules identity tie-break enabled", "yes" if run_stats.get("rules_identity_tiebreak_enabled") else "no", text),
            ("Rules short-name matches", (run_stats.get("rules_enhancement") or {}).get("short_name_matches"), integer),
            ("Rules multiword prefix matches", (run_stats.get("rules_enhancement") or {}).get("multiword_short_name_matches"), integer),
            ("Rules connector multiword prefix matches", run_stats.get("rules_connector_multiword_matches"), integer),
            ("Rules connector ambiguous-prefix abstentions", run_stats.get("rules_connector_ambiguous_prefix_abstentions"), integer),
            ("Rules safe-view matches", (run_stats.get("rules_enhancement") or {}).get("safe_view_matches"), integer),
            ("Rules core-anchor matches", (run_stats.get("rules_enhancement") or {}).get("core_anchor_matches"), integer),
            ("Explicit legal-form recoveries", ((run_stats.get("rules_enhancement") or {}).get("second_pass") or {}).get("explicit_legal_form"), integer),
            ("And/ampersand recoveries", ((run_stats.get("rules_enhancement") or {}).get("second_pass") or {}).get("and_equivalence"), integer),
        ]
    profile_keys = (
        "graph_seconds", "index_seconds", "retrieval_seconds", "retrieval_query_seconds", "char_matrix_seconds",
        "word_matrix_seconds", "shortlist_seconds", "proposal_build_seconds",
        "identity_feature_seconds", "identity_model_seconds", "decision_seconds",
        "rules_decision_seconds",
        "rules_enhancement_seconds",
        "write_seconds", "peak_rss_mb",
    )
    if is_rules:
        profile_keys = ()
    for offset in range(max(len(summary_values), len(profile_keys))):
        row_number = offset + 5
        if offset < len(summary_values):
            label, value, value_format = summary_values[offset]
            summary.write(row_number, 0, label, text)
            if value is None:
                summary.write_blank(row_number, 1, None, value_format)
            else:
                summary.write(row_number, 1, value, value_format)
        if offset < len(profile_keys):
            key = profile_keys[offset]
            summary.write(row_number, 11, key, text)
            if run_stats.get(key) is not None:
                summary.write_number(row_number, 12, float(run_stats[key]), decimal)
    summary.set_column("L:L", 32)
    summary.set_column("M:M", 17)

    split_start = 7 + max(len(summary_values), len(profile_keys))
    _section_band(summary, split_start, 0, 9, "Metrics by dataset split", section)
    split_headers = ["Split", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "F1", "Coverage", "Retrieval recall"]
    summary.write_row(split_start + 1, 0, split_headers, header)
    for row_number, (split, values) in enumerate(metrics["by_split"].items(), start=split_start + 2):
        summary.write(row_number, 0, split, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "f1", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    cohort_start = split_start + 4 + len(metrics["by_split"])
    _section_band(summary, cohort_start, 0, 9, "Held-out review cohorts", section)
    summary.write_row(cohort_start + 1, 0, ["Cohort", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "F1", "Coverage", "Retrieval recall"], header)
    for row_number, (cohort, values) in enumerate(metrics["cohorts"].items(), start=cohort_start + 2):
        summary.write(row_number, 0, cohort, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "f1", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    case_start = cohort_start + 4 + len(metrics["cohorts"])
    _section_band(summary, case_start, 0, 9, "Metrics by case type", section)
    summary.write_row(case_start + 1, 0, ["Case type", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "F1", "Coverage", "Retrieval recall"], header)
    for row_number, (case, values) in enumerate(metrics["by_case"].items(), start=case_start + 2):
        summary.write(row_number, 0, case, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "f1", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    category_start = case_start + 4 + len(metrics["by_case"])
    _section_band(summary, category_start, 0, 9, "Metrics by workbook category", section)
    summary.write_row(category_start + 1, 0, ["Category", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "F1", "Coverage", "Retrieval recall"], header)
    for row_number, (category, values) in enumerate(metrics["by_category"].items(), start=category_start + 2):
        summary.write(row_number, 0, category, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "f1", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    reliability_start = category_start + 4 + len(metrics["by_category"])
    _section_band(summary, reliability_start, 0, 3,
                  "Rules score accuracy by bucket" if is_rules else "Confidence reliability", section)
    summary.write_row(reliability_start + 1, 0,
                      ["Score bucket", "Matches", "Average score", "Observed accuracy"] if is_rules
                      else ["Confidence bucket", "Matches", "Average confidence", "Observed accuracy"], header)
    for row_number, values in enumerate(metrics["confidence_bins"], start=reliability_start + 2):
        summary.write(row_number, 0, values["bucket"], text)
        summary.write(row_number, 1, values["matches"], integer)
        summary.write(row_number, 2, values["average_confidence"], percent)
        summary.write(row_number, 3, values["accuracy"], percent)

    funnel_start = reliability_start + 4 + len(metrics["confidence_bins"])
    _section_band(summary, funnel_start, 0, 1, "Held-out evaluation funnel", section)
    summary.write_row(funnel_start + 1, 0, ["Stage", "Rows"], header)
    for row_number, (stage, value) in enumerate(metrics["funnel"].items(), start=funnel_start + 2):
        summary.write(row_number, 0, stage, text)
        summary.write(row_number, 1, value, integer)

    summary.set_column("A:A", 42)
    summary.set_column("B:C", 14)
    summary.set_column("D:D", 27)
    summary.set_column("E:F", 14)
    summary.set_column("G:I", 16)
    summary.set_column("J:J", 16)
    summary.set_row(split_start + 1, 30)

    detail = workbook.add_worksheet("Detail")
    detail.hide_gridlines(2)
    detail.freeze_panes(1, 1)
    detail_columns = DETAIL_COLUMNS[:4] if is_rules else DETAIL_COLUMNS
    detail.write_row(0, 0, detail_columns, header)
    for row_number, row in enumerate(rows, start=1):
        prediction = row["prediction"]
        is_match = prediction["Decision"] == "MATCH"
        correct_answer = prediction.get("Expected Global Parent Name") or prediction.get("Expected Canonical Name")
        if str(correct_answer or "").upper() == "UNKNOWN":
            correct_answer = None
        result = (
            "Not scored" if not prediction["Scorable"] else
            "Correct" if prediction["Correct"] else
            "Wrong match" if is_match else "No match"
        )
        values = [
            row["source"].get(manifest.get("raw_name_column")),
            prediction.get("Predicted Verified Name") if is_match else "NO_MATCH",
            correct_answer,
            result,
        ]
        if not is_rules:
            values.append(prediction.get("Rules-only prediction"))
        for column, value in enumerate(values):
            if isinstance(value, (list, tuple, set)):
                value = _join(value)
            if isinstance(value, bool):
                detail.write_boolean(row_number, column, value, text)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                chosen = decimal if isinstance(value, float) else integer
                detail.write_number(row_number, column, value, chosen)
            elif value is None:
                detail.write_blank(row_number, column, None, text)
            else:
                detail.write_string(row_number, column, str(value)[:32_767], text)
    detail.autofilter(0, 0, len(rows), len(detail_columns) - 1)
    detail.set_row(0, 30)
    for column, width in enumerate((48, 48, 48, 18) if is_rules else (48, 48, 48, 18, 48)):
        detail.set_column(column, column, width)
    match_format = workbook.add_format({"bg_color": "#E2F0D9", "font_color": "#275D38"})
    no_match_format = workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"})
    wrong_match_format = workbook.add_format({"bg_color": "#FCE8E6", "font_color": "#9C2F24"})
    unscored_format = workbook.add_format({"bg_color": "#F2F4F7", "font_color": "#4B5563"})
    if rows:
        detail.conditional_format(1, 0, len(rows), len(detail_columns) - 1, {
            "type": "formula", "criteria": '=$D2="Correct"', "format": match_format,
        })
        detail.conditional_format(1, 0, len(rows), len(detail_columns) - 1, {
            "type": "formula", "criteria": '=$D2="No match"', "format": no_match_format,
        })
        detail.conditional_format(1, 0, len(rows), len(detail_columns) - 1, {
            "type": "formula", "criteria": '=$D2="Wrong match"', "format": wrong_match_format,
        })
        detail.conditional_format(1, 0, len(rows), len(detail_columns) - 1, {
            "type": "formula", "criteria": '=$D2="Not scored"', "format": unscored_format,
        })
    workbook.close()


def _write_analysis(path: Path, metrics: dict[str, Any], run_stats: dict[str, Any]) -> None:
    overall = metrics["overall"]
    lines = [
        "# Party matching run",
        "",
        f"- All source rows: {int(run_stats.get('records', 0))}",
        f"- Held-out labeled rows: {overall['scorable']}",
        f"- Held-out match precision: {_display_rate(overall['precision'])}",
        f"- Held-out recall: {_display_rate(overall['recall'])}",
        f"- Held-out F1: {_display_rate(overall['f1'])}",
        f"- Held-out coverage: {_display_rate(overall['coverage'])}",
        f"- Held-out retrieval recall: {_display_rate(overall['retrieval_recall'])}",
        f"- Held-out macro entity recall: {_display_rate(metrics.get('macro_entity_recall'))}",
        f"- Matching runtime: {float(run_stats.get('total_seconds', 0)):.2f} seconds",
        f"- Unknown predicted matches (not scored): {metrics['unknown_predictions']['matches']}",
        "",
        "## Stage profile",
        "",
        *[f"- {key}: {value:.2f}" for key, value in run_stats.items()
          if (key.endswith("_seconds") or key.endswith("_rss_mb")) and isinstance(value, (int, float))],
        "",
        "## Limitations",
        "",
        "UNKNOWN rows are included in predictions but excluded from supervised metrics. Results from synthetic data do not establish production quality.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " | ".join(str(item) for item in value)


def _text_chunks(values: list[str]) -> list[str]:
    joined = " | ".join(values)
    return [joined[start:start + 32_000] for start in range(0, len(joined), 32_000)]


def _display_rate(value: float | None) -> str:
    return "n.a." if value is None else f"{value:.2%}"


def _error_bucket(prediction: dict[str, Any]) -> str:
    if not prediction.get("Scorable"):
        return "UNKNOWN_UNSCORED" if prediction.get("Evaluation Status") == "UNKNOWN" else "NOT_EVALUATED"
    if prediction.get("Correct"):
        return "CORRECT"
    if not prediction.get("Expected Target Retrieved"):
        return "RETRIEVAL_MISS"
    if prediction.get("Decision") != "MATCH":
        return "REJECTED_AFTER_RETRIEVAL"
    return "RANKING_ERROR"


def _review_issue(prediction: dict[str, Any]) -> str:
    if not prediction["Scorable"] or prediction["Correct"]:
        return ""
    if prediction["Reason"] == "PREFERRED_SEGMENT_NEAR_CUTOFF":
        return "Preferred part near cutoff"
    if (prediction["Has OBO"] or prediction["Has VIA"]) and prediction["Expected Segment Seen"]:
        return "Wrong connector choice" if prediction["Decision"] == "MATCH" else "Correct part rejected"
    if not prediction["Expected Target Retrieved"]:
        return "Correct party not found"
    return "Wrong party selected" if prediction["Decision"] == "MATCH" else "Candidate rejected"


def _section_band(sheet: Any, row: int, first_column: int, last_column: int, label: str, cell_format: Any) -> None:
    sheet.write(row, first_column, label, cell_format)
    for column in range(first_column + 1, last_column + 1):
        sheet.write_blank(row, column, None, cell_format)


def _case_types(raw_name: str, prediction: dict[str, Any]) -> list[str]:
    normalized = normalize_name(raw_name)
    tokens = normalized.split()
    cases = []
    if prediction.get("Has OBO"):
        cases.append("OBO")
    if prediction.get("Has VIA"):
        cases.append("VIA")
    if raw_name.strip().isupper() and 1 <= len(normalized.replace(" ", "")) <= 8:
        cases.append("ABBREVIATION")
    if len(tokens) == 1 and len(normalized) <= 6:
        cases.append("SHORT_IDENTIFIER")
    if any(token in {"japan", "korea", "uk", "kingdom", "singapore", "brasil", "brazil", "canada", "australia"} for token in tokens):
        cases.append("REGIONAL")
    if len(tokens) >= 3 and len(tokens[0]) <= 3:
        cases.append("SHORT_PREFIX_SHARED_DESCRIPTION")
    if prediction.get("Reason") == "CANDIDATE_COLLISION":
        cases.append("CANDIDATE_COLLISION")
    return cases or ["GENERAL"]


def np_arange(start: float, stop: float, step: float) -> list[float]:
    values, current = [], start
    while current < stop - 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)
