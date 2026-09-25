"""Small Python-only Excel export for the two-snapshot suggestions demo.

The matching logic lives elsewhere. This module only presents saved inference
results, using XlsxWriter already required by the main POC.
"""

from __future__ import annotations

from pathlib import Path

import xlsxwriter


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
