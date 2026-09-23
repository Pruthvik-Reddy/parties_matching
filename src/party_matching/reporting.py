from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from .domain import normalize_name, read_json, read_jsonl, write_json


PREDICTION_COLUMNS = [
    "ADM Party ID", "Source Row", "Dataset Split", "Evaluation Status", "Held-Out",
    "Scorable", "Exclusion Reason",
    "Workbook Canonical ID", "Expected Verified ID", "Expected Canonical Name",
    "Expected Global Parent Name", "Decision", "Correct",
    "Confidence", "Reason", "Event Emitted", "Predicted Verified ID",
    "Predicted Verified Name", "Top Candidate Verified ID", "Top Candidate Verified Name",
    "Matched Member ID", "Matched Member Name",
    "Matched Candidate Name", "Candidate Type", "Candidate Expansion Confidence", "Matched Mention", "Connector",
    "Match Method", "Decision Tier", "Retrieval Sources", "Identity Score",
    "Char TF-IDF Score", "Word TF-IDF Score", "RRF Score", "Char Similarity",
    "Jaro-Winkler", "Levenshtein", "Token Jaccard", "Raw Coverage", "Candidate Coverage",
    "Distinctive Token Conflict", "Digit Conflict", "Cross-Encoder Score",
    "Runner-Up Verified ID", "Runner-Up Score", "Margin", "Graph Path",
    "Retrieved Root IDs", "Retrieved Root Count", "Expected Target Retrieved", "Expected Target Rank",
    "Error Bucket", "Has OBO", "Has VIA", "Parse Warning",
    "Connector Resolution", "Selected Mention Position", "Provisional Verified ID",
    "Provisional Verified Name", "Mention Results",
    "Strategy", "Model Version", "Expansion Version", "Graph Schema Version",
    "Run ID", "System Threshold", "Effective Cutoff",
    "Case Types",
]


def build_reports(prepared_dir: str | Path, output_dir: str | Path, run_stats: dict[str, Any]) -> dict[str, Any]:
    prepared, output = Path(prepared_dir), Path(output_dir)
    manifest = read_json(prepared / "manifest.json", {}) or {}
    labels = {row["adm_party_id"]: row for row in read_jsonl(prepared / "labels.jsonl")}
    decisions = {row["adm_party_id"]: row for row in read_jsonl(output / "decisions.jsonl")}
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


def _metrics(detail_rows: list[dict[str, Any]], run_stats: dict[str, Any]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scorable = [row for row in rows if row["prediction"]["Scorable"]]
        matches = [row for row in rows if row["prediction"]["Decision"] == "MATCH"]
        scored_matches = [row for row in matches if row["prediction"]["Scorable"]]
        correct = [row for row in scored_matches if row["prediction"]["Correct"]]
        retrieved = [row for row in scorable if row["prediction"]["Expected Target Retrieved"]]
        interval = _wilson_interval(len(correct), len(scored_matches)) if scored_matches else (None, None)
        return {
            "rows": len(rows),
            "scorable": len(scorable),
            "matches": len(matches),
            "correct_matches": len(correct),
            "precision": len(correct) / len(scored_matches) if scored_matches else None,
            "precision_ci_95_low": interval[0],
            "precision_ci_95_high": interval[1],
            "recall": len(correct) / len(scorable) if scorable else None,
            # Backward-compatible alias for any existing consumers of metrics.json.
            "correct_match_recall": len(correct) / len(scorable) if scorable else None,
            "coverage": len(scored_matches) / len(scorable) if scorable else None,
            "retrieval_recall": len(retrieved) / len(scorable) if scorable else None,
        }
    held_out = [row for row in detail_rows if row["prediction"]["Held-Out"]]
    all_labeled = [row for row in detail_rows if row["prediction"]["Scorable"]]
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
    return {
        "overall": summarize(held_out),
        "all_labeled": summarize(all_labeled),
        "cohorts": cohorts,
        "by_split": {
            split: summarize([row for row in detail_rows if row["prediction"]["Dataset Split"] == split])
            for split in split_names
        },
        "by_case": by_case,
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
) -> None:
    workbook = xlsxwriter.Workbook(path, {"constant_memory": True, "strings_to_formulas": False, "strings_to_urls": False})
    workbook.set_properties({"title": "Party matching predictions", "subject": "POC evaluation output"})
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

    summary = workbook.add_worksheet("Summary")
    summary.hide_gridlines(2)
    summary.set_tab_color("#1F4E78")
    summary.write("A2", "Party matching evaluation", title)
    _section_band(summary, 3, 0, 1, "Run summary", section)
    _section_band(summary, 3, 11, 12, "Matching stage profile", section)
    summary.write_row("A5", ["Metric", "Value"], header)
    summary.write_row(4, 11, ["Stage", "Seconds / MB"], header)
    overall = metrics["overall"]
    summary_values = [
        ("All source rows", run_stats.get("records"), integer),
        ("Held-out labeled rows", overall["scorable"], integer),
        ("Held-out matches", overall["matches"], integer),
        ("Held-out correct matches", overall["correct_matches"], integer),
        ("Events written", run_stats.get("events"), integer),
        ("Held-out precision", overall["precision"], percent),
        ("Precision 95% CI low", overall["precision_ci_95_low"], percent),
        ("Precision 95% CI high", overall["precision_ci_95_high"], percent),
        ("Held-out recall", overall["recall"], percent),
        ("Held-out macro entity recall", metrics.get("macro_entity_recall"), percent),
        ("Held-out coverage", overall["coverage"], percent),
        ("Held-out retrieval recall", overall["retrieval_recall"], percent),
        ("Unknown rows (not scored)", metrics["unknown_predictions"]["rows"], integer),
        ("Unknown predicted matches", metrics["unknown_predictions"]["matches"], integer),
        ("Matching runtime (seconds)", run_stats.get("total_seconds"), decimal),
        ("System threshold", run_stats.get("system_threshold"), decimal),
        ("Effective cutoff", run_stats.get("effective_cutoff"), decimal),
    ]
    profile_keys = (
        "graph_seconds", "index_seconds", "retrieval_seconds", "retrieval_query_seconds", "char_matrix_seconds",
        "word_matrix_seconds", "shortlist_seconds", "proposal_build_seconds",
        "identity_feature_seconds", "identity_model_seconds", "decision_seconds",
        "write_seconds", "peak_rss_mb",
    )
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

    split_start = 24
    _section_band(summary, split_start, 0, 8, "Metrics by dataset split", section)
    split_headers = ["Split", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "Coverage", "Retrieval recall"]
    summary.write_row(split_start + 1, 0, split_headers, header)
    for row_number, (split, values) in enumerate(metrics["by_split"].items(), start=split_start + 2):
        summary.write(row_number, 0, split, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    cohort_start = split_start + 4 + len(metrics["by_split"])
    _section_band(summary, cohort_start, 0, 8, "Held-out connector cohorts", section)
    summary.write_row(cohort_start + 1, 0, ["Cohort", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "Coverage", "Retrieval recall"], header)
    for row_number, (cohort, values) in enumerate(metrics["cohorts"].items(), start=cohort_start + 2):
        summary.write(row_number, 0, cohort, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    case_start = cohort_start + 4 + len(metrics["cohorts"])
    _section_band(summary, case_start, 0, 8, "Metrics by case type", section)
    summary.write_row(case_start + 1, 0, ["Case type", "Rows", "Scorable", "Matches", "Correct", "Precision", "Recall", "Coverage", "Retrieval recall"], header)
    for row_number, (case, values) in enumerate(metrics["by_case"].items(), start=case_start + 2):
        summary.write(row_number, 0, case, text)
        for column, key in enumerate(("rows", "scorable", "matches", "correct_matches"), start=1):
            summary.write(row_number, column, values[key], integer)
        for column, key in enumerate(("precision", "recall", "coverage", "retrieval_recall"), start=5):
            if values[key] is not None:
                summary.write(row_number, column, values[key], percent)

    reliability_start = case_start + 4 + len(metrics["by_case"])
    _section_band(summary, reliability_start, 0, 3, "Confidence reliability", section)
    summary.write_row(reliability_start + 1, 0, ["Confidence bucket", "Matches", "Average confidence", "Observed accuracy"], header)
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

    canonical_start = funnel_start + 4 + len(metrics["funnel"])
    _section_band(summary, canonical_start, 0, 9, "Results by expected canonical name", section)
    canonical_headers = ["Expected canonical name", "Rows", "Matches", "Correct", "Precision"]
    summary.write_row(canonical_start + 1, 0, canonical_headers, header)
    summary.write(canonical_start + 1, 9, "Raw aliases", header)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        canonical = str(row["prediction"].get("Expected Canonical Name") or "UNKNOWN")
        grouped[canonical].append(row)
    for row_number, canonical in enumerate(sorted(grouped, key=str.casefold), start=canonical_start + 2):
        group = grouped[canonical]
        matched = [row for row in group if row["prediction"]["Decision"] == "MATCH"]
        correct = [row for row in matched if row["prediction"]["Correct"]]
        aliases = " | ".join(
            str(value) for value in (row["source"].get(manifest.get("raw_name_column")) for row in group)
            if value is not None and str(value).strip()
        )
        if len(aliases) > 32_000:
            aliases = aliases[:31_970] + " ... [truncated; Detail is authoritative]"
        summary.write(row_number, 0, canonical, text)
        summary.write(row_number, 1, len(group), integer)
        summary.write(row_number, 2, len(matched), integer)
        summary.write(row_number, 3, len(correct), integer)
        if matched:
            summary.write(row_number, 4, len(correct) / len(matched), percent)
        summary.write(row_number, 9, aliases, text)
    summary.set_column("A:A", 42)
    summary.set_column("B:E", 14)
    summary.set_column("F:I", 16)
    summary.set_column("J:J", 80)
    summary.set_row(split_start + 1, 30)

    detail = workbook.add_worksheet("Detail")
    detail.hide_gridlines(2)
    detail.freeze_panes(1, 3)
    raw_column, label_column = manifest.get("raw_name_column"), manifest.get("canonical_name_column")
    source_columns = [column for column in manifest.get("source_columns", []) if column not in {raw_column, label_column}]
    first_columns = ["Raw Name", "Label", "Prediction", "Expected Global Parent", "Correct", "Decision"]
    prediction_columns = [column for column in PREDICTION_COLUMNS if column not in {"Correct", "Decision"}]
    columns = first_columns + source_columns + prediction_columns
    detail.write_row(0, 0, columns, header)
    for row_number, row in enumerate(rows, start=1):
        prediction = row["prediction"]
        values = [row["source"].get(raw_column), row["source"].get(label_column),
                  prediction.get("Predicted Verified Name") or "NO_MATCH",
                  prediction.get("Expected Global Parent Name"), prediction.get("Correct"),
                  prediction.get("Decision")]
        values.extend(row["source"].get(column) for column in source_columns)
        values.extend(prediction.get(column) for column in prediction_columns)
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
    detail.autofilter(0, 0, len(rows), len(columns) - 1)
    detail.set_row(0, 30)
    for column, name in enumerate(columns):
        width = 18
        lowered = name.casefold()
        if "name" in lowered or "path" in lowered or "sources" in lowered or "reason" in lowered:
            width = 30
        elif "id" in lowered:
            width = 38
        detail.set_column(column, column, width)
    decision_column = columns.index("Decision")
    correct_column = columns.index("Correct")
    match_format = workbook.add_format({"bg_color": "#E2F0D9", "font_color": "#275D38"})
    no_match_format = workbook.add_format({"bg_color": "#FCE8E6", "font_color": "#9C2F24"})
    if rows:
        detail.conditional_format(1, decision_column, len(rows), decision_column, {
            "type": "cell", "criteria": "==", "value": '"MATCH"', "format": match_format,
        })
        detail.conditional_format(1, decision_column, len(rows), decision_column, {
            "type": "cell", "criteria": "==", "value": '"NO_MATCH"', "format": no_match_format,
        })
        scorable_column = columns.index("Scorable")
        scorable_ref = xl_col_to_name(scorable_column)
        correct_ref = xl_col_to_name(correct_column)
        detail.conditional_format(1, correct_column, len(rows), correct_column, {
            "type": "formula",
            "criteria": f"=AND(${scorable_ref}2=TRUE,${correct_ref}2=FALSE)",
            "format": no_match_format,
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
