"""Small Python-only Excel export for the two-snapshot suggestions demo.

The matching logic lives elsewhere. This module only presents saved inference
results, using XlsxWriter already required by the main POC.
"""

from __future__ import annotations

from pathlib import Path

import xlsxwriter


EXCEL_DATA_ROWS_PER_SHEET = 1_048_575


def _table(book, title: str, headers: list[str], rows: list[dict], keys: list[str],
           widths: list[int], percent_keys: set[str] | None = None,
           percent_row_numbers: set[int] | None = None) -> None:
    sheet = book.add_worksheet(title)
    heading = book.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D",
                               "text_wrap": True, "valign": "vcenter"})
    percent = book.add_format({"num_format": "0.0%"})
    number = book.add_format({"num_format": "#,##0"})
    decimal = book.add_format({"num_format": "0.000"})
    sheet.set_row(0, 30)
    for col, (header, width) in enumerate(zip(headers, widths)):
        sheet.write(0, col, header, heading)
        sheet.set_column(col, col, width)
    for row_number, record in enumerate(rows, 1):
        for col, key in enumerate(keys):
            value = record.get(key, "")
            if value is None:
                value = ""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                style = percent if (key in (percent_keys or set()) or
                                    row_number in (percent_row_numbers or set())) else (
                    number if isinstance(value, int) else decimal)
                sheet.write_number(row_number, col, value, style)
            else:
                sheet.write_string(row_number, col, str(value))
    sheet.freeze_panes(1, 1)
    sheet.autofilter(0, 0, max(1, len(rows)), len(headers) - 1)


def export_snapshot(folder: Path, result: tuple[list[dict], list[dict], dict],
                    changes: list[dict] | None = None,
                    scenario: dict | None = None) -> Path:
    """Write one compact workbook; no Node runtime or formulas are required."""
    strong, review, summary = result
    path = folder / "suggestions_current.xlsx"
    book = xlsxwriter.Workbook(str(path), {"constant_memory": True,
                                           "strings_to_formulas": False,
                                           "strings_to_urls": False})
    try:
        _table(book, "Verified parties",
               ["Verified party", "Strong suggestions", "Review candidates", "Verified party ID"],
               summary.get("verified_parties", []),
               ["verified_party", "strong_count", "review_count", "verified_party_id"],
               [48, 20, 20, 40])
        _table(book, "Strong suggestions",
               ["Verified party", "Group count", "Unverified party", "Rules score",
                "Match method", "Decision tier", "Case", "Matched segment",
                "Connector decision", "Source row", "Verified ID", "Unverified ID"],
               strong,
               ["verified_party", "verified_strong_count", "unverified_party", "rules_score",
                "match_method", "decision_tier", "name_case", "matched_segment",
                "connector_resolution", "source_row", "verified_party_id", "unverified_party_id"],
               [48, 14, 62, 14, 40, 32, 14, 48, 22, 14, 40, 40])
        _table(book, "Review candidates",
               ["Verified party", "Group count", "Unverified party", "Identity score",
                "Lexical score", "Why rejected", "Candidate name", "Case",
                "Matched segment", "Connector decision", "Source row", "Verified ID", "Unverified ID"],
               review,
               ["verified_party", "verified_review_count", "unverified_party", "identity_score",
                "lexical_score", "decision_reason", "candidate_name", "name_case",
                "matched_segment", "connector_resolution", "source_row",
                "verified_party_id", "unverified_party_id"],
               [48, 14, 62, 14, 14, 28, 48, 14, 48, 22, 14, 40, 40])
        if changes is not None:
            _table(book, "Changes",
                   ["Change", "Unverified party", "Case", "Initial status",
                    "Initial verified party", "Initial review candidates", "Expanded status",
                    "Expanded verified party", "Expanded review candidates", "Unverified ID"],
                   changes,
                   ["change", "unverified_party", "name_case", "initial_status",
                    "initial_verified_party", "initial_review_candidates", "expanded_status",
                    "expanded_verified_party", "expanded_review_candidates", "unverified_party_id"],
                   [20, 62, 14, 18, 48, 58, 18, 48, 58, 40])
        if scenario is not None:
            coverage_before = scenario["initial_coverage"]
            coverage_after = scenario["expanded_coverage"]
            details = [
                {"item": "Scenario", "value": "Simulated earlier verified catalog"},
                {"item": "Selected unverified names", "value": scenario["selected_unverified_names"]},
                {"item": "Initial strong suggestions", "value": coverage_before["strong_names"]},
                {"item": "Expanded strong suggestions", "value": coverage_after["strong_names"]},
                {"item": "Initial review-only names", "value": coverage_before["review_only_names"]},
                {"item": "Expanded review-only names", "value": coverage_after["review_only_names"]},
                {"item": "Initial suggestion coverage", "value": coverage_before["suggestion_coverage"]},
                {"item": "Expanded suggestion coverage", "value": coverage_after["suggestion_coverage"]},
                {"item": "Caution", "value": "Simulated order; review suggestions are not accepted matches"},
            ]
            _table(book, "Scenario", ["Measure", "Value"], details,
                   ["item", "value"], [35, 78], percent_row_numbers={7, 8})
            sheet = book.get_worksheet_by_name("Scenario")
            heading = book.add_format({"bold": True, "font_color": "white", "bg_color": "#17365D"})
            start = 12
            for col, label in enumerate(("Selected representative", "Shared token", "Verified names",
                                         "Eligible raw names")):
                sheet.write(start, col, label, heading)
            for index, family in enumerate(scenario.get("families", []), start + 1):
                sheet.write(index, 0, family["representative"])
                sheet.write(index, 1, family["anchor"])
                sheet.write_number(index, 2, len(family["verified_names"]))
                sheet.write_number(index, 3, family["eligible_unverified_with_token"])
            start += len(scenario.get("families", [])) + 3
            sheet.write(start, 0, "Verified names withheld initially", heading)
            for index, party in enumerate(scenario.get("withheld_initially", []), start + 1):
                sheet.write(index, 0, party["partyName"])
                sheet.write(index, 1, party["partyId"])
            sheet.set_column(2, 2, 18)
            sheet.set_column(3, 3, 20)
    finally:
        book.close()
    return path


def export_family_snapshot(folder: Path, result: dict) -> Path:
    """Display saved pair outcomes; labels remain in a separate audit tab.

    The established Python-only exporter is retained for the work-laptop demo.
    No workbook formula participates in retrieval, scoring, or classification.
    """
    path = folder / "suggestions_current.xlsx"
    book = xlsxwriter.Workbook(str(path), {"constant_memory": True,
                                           "strings_to_formulas": False,
                                           "strings_to_urls": False})
    try:
        parties = result["parties"]
        _table(book, "Verified parties",
               ["Verified party", "Final rollup", "Strong", "Review", "Dropped", "Known labels",
                "Not retrieved", "Family-gate exclusions", "Held-out candidate recall", "Held-out strong precision",
                "Verified ID"], parties,
               ["verified_party", "rolled_into", "strong", "review", "dropped", "known_labels",
                "known_label_index_misses", "known_label_gate_exclusions", "held_out_candidate_recall",
                "held_out_strong_precision", "verified_party_id"],
               [48, 48, 14, 14, 14, 16, 20, 24, 24, 24, 40],
               percent_keys={"held_out_candidate_recall", "held_out_strong_precision"})
        detail_headers = ["Verified party", "Unverified party", "Reason", "Rules score",
                          "Lexical score", "Family similarity", "Retrieval source",
                          "Matched segment", "Case", "Matched verified party", "Assigned elsewhere to", "Source row",
                          "Verified ID", "Unverified ID"]
        detail_keys = ["verified_party", "unverified_party", "reason", "rules_score",
                       "lexical_score", "family_similarity", "retrieval_source",
                       "matched_segment", "name_case", "matched_verified_party", "assigned_to", "source_row",
                       "verified_party_id", "unverified_party_id"]
        widths = [48, 62, 32, 15, 15, 17, 22, 48, 14, 48, 48, 14, 40, 40]
        for title, key in (("Strong suggestions", "strong"),
                           ("Review candidates", "review"),
                           ("Dropped candidates", "dropped")):
            for start in range(0, max(1, len(result[key])), EXCEL_DATA_ROWS_PER_SHEET):
                number = start // EXCEL_DATA_ROWS_PER_SHEET + 1
                sheet_title = title if number == 1 else f"{title} {number}"
                _table(book, sheet_title, detail_headers,
                       result[key][start:start + EXCEL_DATA_ROWS_PER_SHEET], detail_keys, widths)
        gap_headers = ["Verified party", "Unverified party", "Expected party", "Split",
                       "Reason", "Source row", "Verified ID", "Unverified ID"]
        gap_keys = ["verified_party", "unverified_party", "expected_party", "split",
                    "reason", "source_row", "verified_party_id", "unverified_party_id"]
        for start in range(0, max(1, len(result["misses"])), EXCEL_DATA_ROWS_PER_SHEET):
            number = start // EXCEL_DATA_ROWS_PER_SHEET + 1
            sheet_title = "Known-label gaps" if number == 1 else f"Known-label gaps {number}"
            _table(book, sheet_title, gap_headers,
                   result["misses"][start:start + EXCEL_DATA_ROWS_PER_SHEET],
                   gap_keys, [48, 62, 48, 18, 34, 14, 40, 40])
        stats = result["metrics"]
        stat_rows = [
            {"metric": "Stage", "value": result["stage"]},
            {"metric": "Eligible unverified rows", "value": stats["eligible_unverified_rows"]},
            {"metric": "Selected verified parties", "value": stats["selected_verified_parties"]},
            {"metric": "Candidate pairs", "value": stats["candidate_pairs"]},
            {"metric": "Strong pairs", "value": stats["strong_pairs"]},
            {"metric": "Review pairs", "value": stats["review_pairs"]},
            {"metric": "Dropped pairs", "value": stats["dropped_pairs"]},
            {"metric": "Known-label candidate gaps", "value": stats["known_label_misses"]},
            {"metric": "Not retrieved for party", "value": stats["known_label_index_misses"]},
            {"metric": "Indexed but failed family gate", "value": stats["known_label_gate_exclusions"]},
            {"metric": "Stage runtime (seconds)", "value": stats["stage_seconds"]},
            {"metric": "Scope", "value": stats["scope"]},
            {"metric": "Label audit", "value": stats["label_scope"]},
        ]
        _table(book, "Stats", ["Metric", "Value"], stat_rows,
               ["metric", "value"], [37, 105])
    finally:
        book.close()
    return path
