from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_col_to_name

from .domain import normalize_name, read_json, read_jsonl, write_json


PREDICTION_COLUMNS = [
    "ADM Party ID", "Source Row", "Dataset Split", "Scorable", "Exclusion Reason",
    "Expected Verified ID", "Expected Canonical Name", "Decision", "Correct",
    "Confidence", "Reason", "Event Emitted", "Predicted Verified ID",
    "Predicted Verified Name", "Top Candidate Verified ID", "Top Candidate Verified Name",
    "Matched Member ID", "Matched Member Name",
    "Matched Candidate Name", "Candidate Type", "Candidate Expansion Confidence", "Matched Mention", "Connector",
    "Match Method", "Retrieval Sources", "Identity Score", "Cross-Encoder Score",
    "Runner-Up Verified ID", "Runner-Up Score", "Margin", "Graph Path",
    "Retrieved Root IDs", "Expected Target Retrieved", "Parse Warning",
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
    detail_rows = []
    for source_row in source_rows:
        adm_id = source_row["adm_party_id"]
        if adm_id not in decisions:
            continue
        label = labels[adm_id]
        decision = decisions[adm_id]
        expected = label.get("expected_party_id")
        candidate_id = decision.get("verified_party_id")
        candidate_name = decision.get("verified_party_name")
        is_match = decision.get("decision") == "MATCH"
        predicted = candidate_id if is_match else None
        correct = (predicted == expected) if label.get("scorable") and is_match else (False if label.get("scorable") else None)
        expected_retrieved = expected in set(decision.get("retrieved_root_ids") or []) if expected else None
        prediction = {
            "ADM Party ID": adm_id,
            "Source Row": source_row["source_row"],
            "Dataset Split": label.get("split"),
            "Scorable": bool(label.get("scorable")),
            "Exclusion Reason": label.get("exclusion_reason"),
            "Expected Verified ID": expected,
            "Expected Canonical Name": label.get("expected_canonical_name"),
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
            "Retrieval Sources": _join(decision.get("retrieval_sources")),
            "Identity Score": decision.get("identity_score"),
            "Cross-Encoder Score": decision.get("cross_encoder_score"),
            "Runner-Up Verified ID": decision.get("runner_up_party_id"),
            "Runner-Up Score": decision.get("runner_up_score"),
            "Margin": decision.get("margin"),
            "Graph Path": _join(decision.get("graph_path")),
            "Retrieved Root IDs": _join(decision.get("retrieved_root_ids")),
            "Expected Target Retrieved": expected_retrieved,
            "Parse Warning": decision.get("parse_warning"),
            "Strategy": run_stats.get("strategy"),
            "Model Version": run_stats.get("model_version"),
            "Expansion Version": run_stats.get("expansion_version"),
            "Graph Schema Version": run_stats.get("graph_schema_version"),
            "Run ID": run_stats.get("run_id"),
            "System Threshold": run_stats.get("system_threshold"),
            "Effective Cutoff": run_stats.get("effective_cutoff"),
        }
        prediction["Case Types"] = _join(_case_types(str(source_row["source"].get(manifest.get("raw_name_column")) or ""), prediction))
        detail_rows.append({"source": source_row["source"], "prediction": prediction})
    metrics = _metrics(detail_rows, run_stats)
    write_json(output / "run_metrics.json", metrics)
    _write_analysis(output / "analysis.md", metrics, run_stats)
    _write_workbook(output / "predictions.xlsx", detail_rows, manifest, metrics, run_stats)
    return metrics


def _metrics(detail_rows: list[dict[str, Any]], run_stats: dict[str, Any]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scorable = [row for row in rows if row["prediction"]["Scorable"]]
        matches = [row for row in scorable if row["prediction"]["Decision"] == "MATCH"]
        correct = [row for row in matches if row["prediction"]["Correct"]]
        retrieved = [row for row in scorable if row["prediction"]["Expected Target Retrieved"]]
        interval = _wilson_interval(len(correct), len(matches)) if matches else (None, None)
        return {
            "rows": len(rows),
            "scorable": len(scorable),
            "matches": len(matches),
            "correct_matches": len(correct),
            "precision": len(correct) / len(matches) if matches else None,
            "precision_ci_95_low": interval[0],
            "precision_ci_95_high": interval[1],
            "recall": len(correct) / len(scorable) if scorable else None,
            # Backward-compatible alias for any existing consumers of metrics.json.
            "correct_match_recall": len(correct) / len(scorable) if scorable else None,
            "coverage": len(matches) / len(scorable) if scorable else None,
            "retrieval_recall": len(retrieved) / len(scorable) if scorable else None,
        }
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
            row for row in detail_rows
            if case in str(row["prediction"].get("Case Types") or "").split(" | ")
        ])
    bins = []
    for lower in np_arange(0.0, 1.0, 0.1):
        upper = round(lower + 0.1, 10)
        bucket = [
            row for row in detail_rows
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
    return {
        "overall": summarize(detail_rows),
        "by_split": {
            split: summarize([row for row in detail_rows if row["prediction"]["Dataset Split"] == split])
            for split in split_names
        },
        "by_case": by_case,
        "confidence_bins": bins,
        "no_match_reasons": dict(Counter(
            row["prediction"]["Reason"] for row in detail_rows
            if row["prediction"]["Decision"] == "NO_MATCH"
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
    summary.write_row("A5", ["Metric", "Value"], header)
    overall = metrics["overall"]
    summary_values = [
        ("Source rows", overall["rows"], integer),
        ("Scorable rows", overall["scorable"], integer),
        ("Matches", overall["matches"], integer),
        ("Correct matches", overall["correct_matches"], integer),
        ("Events written", run_stats.get("events"), integer),
        ("Precision", overall["precision"], percent),
        ("Precision 95% CI low", overall["precision_ci_95_low"], percent),
        ("Precision 95% CI high", overall["precision_ci_95_high"], percent),
        ("Recall", overall["recall"], percent),
        ("Coverage", overall["coverage"], percent),
        ("Retrieval recall", overall["retrieval_recall"], percent),
        ("Matching runtime (seconds)", run_stats.get("total_seconds"), decimal),
        ("System threshold", run_stats.get("system_threshold"), decimal),
        ("Effective cutoff", run_stats.get("effective_cutoff"), decimal),
    ]
    for row_number, (label, value, value_format) in enumerate(summary_values, start=5):
        summary.write(row_number, 0, label, text)
        if value is None:
            summary.write_blank(row_number, 1, None, value_format)
        else:
            summary.write(row_number, 1, value, value_format)

    split_start = 20
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

    case_start = split_start + 4 + len(metrics["by_split"])
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

    canonical_start = reliability_start + 4 + len(metrics["confidence_bins"])
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
    detail.freeze_panes(1, 2)
    source_columns = list(manifest.get("source_columns", []))
    columns = source_columns + PREDICTION_COLUMNS
    detail.write_row(0, 0, columns, header)
    for row_number, row in enumerate(rows, start=1):
        values = [row["source"].get(column) for column in source_columns]
        values.extend(row["prediction"].get(column) for column in PREDICTION_COLUMNS)
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
        f"- Rows: {overall['rows']}",
        f"- Scorable rows: {overall['scorable']}",
        f"- Match precision: {_display_rate(overall['precision'])}",
        f"- Recall: {_display_rate(overall['recall'])}",
        f"- Coverage: {_display_rate(overall['coverage'])}",
        f"- Retrieval recall: {_display_rate(overall['retrieval_recall'])}",
        f"- Matching runtime: {float(run_stats.get('total_seconds', 0)):.2f} seconds",
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


def _section_band(sheet: Any, row: int, first_column: int, last_column: int, label: str, cell_format: Any) -> None:
    sheet.write(row, first_column, label, cell_format)
    for column in range(first_column + 1, last_column + 1):
        sheet.write_blank(row, column, None, cell_format)


def _case_types(raw_name: str, prediction: dict[str, Any]) -> list[str]:
    normalized = normalize_name(raw_name)
    tokens = normalized.split()
    cases = []
    upper = raw_name.upper()
    if " OBO " in f" {upper} ":
        cases.append("OBO")
    if " VIA " in f" {upper} ":
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
