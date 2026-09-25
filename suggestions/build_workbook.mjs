// Current-state workbook only. The POC and legacy suggestion commands both
// use this builder; neither writes historical assignments into the workbook.
import fs from "node:fs/promises";
import path from "node:path";
import {SpreadsheetFile, Workbook} from "@oai/artifact-tool";

const [outputDir, payloadPath] = process.argv.slice(2);
if (!outputDir || !payloadPath) throw new Error("Expected output directory and payload path");
const {suggestions, ambiguous, summary, changes, scenario} = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const poc = summary.workflow === "poc_rules_suggestions";
const countKey = poc ? "strong_count" : "suggestion_count";
if (suggestions.length !== summary.verified_parties.reduce((n, p) => n + p[countKey], 0)) {
  throw new Error("Verified-party counts do not equal suggestion detail rows");
}
if (poc && ambiguous.length !== summary.verified_parties.reduce((n, p) => n + p.review_count, 0)) {
  throw new Error("Verified-party review counts do not equal review detail rows");
}
if (Math.max(suggestions.length, ambiguous.length, summary.verified_parties.length,
             changes?.length ?? 0, scenario?.withheld_initially?.length ?? 0) > 1048570) {
  throw new Error("One sheet exceeds Excel's row limit; no rows will be silently omitted");
}
const workbook = Workbook.create();
const navy = "#17365D", ink = "#172B4D", muted = "#52627A";

function setup(sheet, title, caveat, headers, widths) {
  sheet.showGridLines = false;
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = {name: "Arial", size: 15, bold: true, color: ink};
  sheet.getRange("A3").values = [[caveat]];
  sheet.getRange("A3").format.font = {name: "Arial", size: 10, color: muted};
  sheet.getRangeByIndexes(4, 0, 1, headers.length).values = [headers];
  sheet.getRangeByIndexes(4, 0, 1, headers.length).format = {
    fill: navy, font: {name: "Arial", size: 10, bold: true, color: "#FFFFFF"},
    rowHeight: 29, verticalAlignment: "center",
  };
  for (let i = 0; i < widths.length; i++) {
    const letter = String.fromCharCode(65 + i);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widths[i];
  }
  sheet.freezePanes.freezeRows(5);
}

function writeRows(sheet, rows, columns) {
  for (let offset = 0; offset < rows.length; offset += 2000) {
    const block = rows.slice(offset, offset + 2000).map(row => columns.map(key => row[key] ?? ""));
    sheet.getRangeByIndexes(5 + offset, 0, block.length, columns.length).values = block;
  }
}

const summaries = workbook.worksheets.add("Verified parties");
if (poc) {
  setup(summaries, `${summary.snapshot_stage ? summary.snapshot_stage + " - " : ""}verified-party suggestion counts`,
    "Strong = accepted by scoped POC rules. Review candidates are not accepted matches.",
    ["Verified party", "Strong suggestions", "Review candidates", "Verified party ID"],
    [50, 20, 20, 43]);
  writeRows(summaries, summary.verified_parties.map(row => ({
    ...row, strong_count: Number(row.strong_count), review_count: Number(row.review_count),
  })), ["verified_party", "strong_count", "review_count", "verified_party_id"]);
  if (summary.verified_parties.length) {
    summaries.getRange(`B6:C${5 + summary.verified_parties.length}`).setNumberFormat("#,##0");
  }
} else {
  setup(summaries, "Verified parties and current suggestion counts",
    "Every count is provisional. Related names do not establish identity or ownership.",
    ["Verified party", "Suggestion count", "Verified party ID"], [47, 20, 43]);
  writeRows(summaries, summary.verified_parties.map(row => ({
    ...row, suggestion_count: Number(row.suggestion_count),
  })), ["verified_party", "suggestion_count", "verified_party_id"]);
  if (summary.verified_parties.length) {
    summaries.getRange(`B6:B${5 + summary.verified_parties.length}`).setNumberFormat("#,##0");
  }
}

const detailName = poc ? "Strong suggestions" : "Suggestions";
const detail = workbook.worksheets.add(detailName);
if (poc) {
  setup(detail, `${summary.snapshot_stage ? summary.snapshot_stage + " - " : ""}strong suggestions`,
    "Includes plain, OBO and VIA names. Rules score is not a calibrated probability.",
    ["Verified party", "Group count", "Unverified party", "Rules score",
     "Match method", "Decision tier", "Case", "Matched segment", "Connector decision",
     "Source row", "Verified ID", "Unverified ID"],
    [47, 15, 58, 17, 43, 42, 14, 48, 24, 14, 42, 42]);
  writeRows(detail, suggestions.map(row => ({
    ...row, verified_strong_count: Number(row.verified_strong_count),
    rules_score: Number(row.rules_score), source_row: Number(row.source_row) || row.source_row,
  })), ["verified_party", "verified_strong_count", "unverified_party", "rules_score",
       "match_method", "decision_tier", "name_case", "matched_segment",
       "connector_resolution", "source_row", "verified_party_id", "unverified_party_id"]);
  if (suggestions.length) {
    detail.getRange(`B6:B${5 + suggestions.length}`).setNumberFormat("#,##0");
    detail.getRange(`D6:D${5 + suggestions.length}`).setNumberFormat("0.000");
  }
} else {
  setup(detail, "All current provisional suggestions",
    "Grouped by verified party. A new verified list causes all eligible names to be re-scored.",
    ["Verified party", "Count for verified", "Unverified party", "Name evidence",
     "Source row", "Verified ID", "Unverified ID", "Status"],
    [45, 19, 56, 57, 13, 40, 40, 17]);
  writeRows(detail, suggestions.map(row => ({
    ...row, verified_suggestion_count: Number(row.verified_suggestion_count),
    source_row: Number(row.source_row) || row.source_row,
  })), ["verified_party", "verified_suggestion_count", "unverified_party",
       "evidence", "source_row", "verified_party_id", "unverified_party_id", "status"]);
  if (suggestions.length) detail.getRange(`B6:B${5 + suggestions.length}`).setNumberFormat("#,##0");
}

const reviewName = poc ? "Review candidates" : "Ambiguous review";
const review = workbook.worksheets.add(reviewName);
if (poc) {
  setup(review, `${summary.snapshot_stage ? summary.snapshot_stage + " - " : ""}review candidates`,
    "Not accepted matches. OBO/VIA reviews use only the preferred connector segment.",
    ["Verified party", "Group count", "Unverified party", "Identity score",
     "Lexical score", "Why rejected", "Candidate name", "Case", "Matched segment",
     "Connector decision", "Source row", "Verified ID", "Unverified ID"],
    [46, 15, 58, 17, 17, 37, 46, 14, 48, 24, 14, 42, 42]);
  writeRows(review, ambiguous.map(row => ({
    ...row, verified_review_count: Number(row.verified_review_count),
    identity_score: Number(row.identity_score), lexical_score: Number(row.lexical_score),
    source_row: Number(row.source_row) || row.source_row,
  })), ["verified_party", "verified_review_count", "unverified_party", "identity_score",
       "lexical_score", "decision_reason", "candidate_name", "name_case",
       "matched_segment", "connector_resolution", "source_row",
       "verified_party_id", "unverified_party_id"]);
  if (ambiguous.length) {
    review.getRange(`B6:B${5 + ambiguous.length}`).setNumberFormat("#,##0");
    review.getRange(`D6:E${5 + ambiguous.length}`).setNumberFormat("0.000");
  }
} else {
  setup(review, "Ambiguous names — no party assigned",
    "Competing verified names have equal evidence; these rows are not included in suggestion counts.",
    ["Unverified party", "Possible verified parties", "Reason", "Source row", "Unverified ID"],
    [55, 72, 55, 13, 43]);
  writeRows(review, ambiguous.map(row => ({
    ...row, source_row: Number(row.source_row) || row.source_row,
  })), ["unverified_party", "possible_verified_parties", "reason", "source_row", "unverified_party_id"]);
}

if (poc && changes !== null && changes !== undefined) {
  const movement = workbook.worksheets.add("Changes");
  setup(movement, "Changes between verified-party snapshots",
    "Only changed names appear. Strong suggestions are accepted rules decisions; review candidates are not matches.",
    ["Change", "Unverified party", "Case", "Initial status", "Initial verified party",
     "Initial review candidates", "Expanded status", "Expanded verified party",
     "Expanded review candidates", "Unverified ID"],
    [24, 60, 14, 18, 50, 58, 18, 50, 58, 42]);
  writeRows(movement, changes,
    ["change", "unverified_party", "name_case", "initial_status", "initial_verified_party",
     "initial_review_candidates", "expanded_status", "expanded_verified_party",
     "expanded_review_candidates", "unverified_party_id"]);
}

if (poc && scenario) {
  const scope = workbook.worksheets.add("Scenario");
  setup(scope, "Simulated verified-list change",
    "This is a name-based demo, not actual onboarding history. Review withheld verified parties below.",
    ["Verified party withheld initially", "Verified party ID"], [56, 43]);
  writeRows(scope, scenario.withheld_initially.map(row => ({
    verified_party: row.partyName, verified_party_id: row.partyId,
  })), ["verified_party", "verified_party_id"]);
  scope.getRange("D5:E17").values = [
    ["Scenario detail", "Value"],
    ["Initial representative", scenario.representative],
    ["Family token", scenario.family_anchor],
    ["Unverified names scored", Number(scenario.selected_unverified_names)],
    ["Input selection", scenario.input_selection],
    ["Added only in expanded", scenario.added_only_in_expanded.join(" / ")],
    ["Same-root family parties", scenario.same_root_as_representative_in_expanded_graph.join(" / ")],
    ["Initial strong names", Number(scenario.initial_coverage.strong_names)],
    ["Initial review-only names", Number(scenario.initial_coverage.review_only_names)],
    ["Initial suggestion coverage", Number(scenario.initial_coverage.suggestion_coverage)],
    ["Expanded strong names", Number(scenario.expanded_coverage.strong_names)],
    ["Expanded review-only names", Number(scenario.expanded_coverage.review_only_names)],
    ["Expanded suggestion coverage", Number(scenario.expanded_coverage.suggestion_coverage)],
  ];
  scope.getRange("D5:E5").format = {
    fill: navy, font: {name: "Arial", size: 10, bold: true, color: "#FFFFFF"},
    rowHeight: 29, verticalAlignment: "center",
  };
  scope.getRange("D:D").format.columnWidth = 30;
  scope.getRange("E:E").format.columnWidth = 68;
  scope.getRange("E8").setNumberFormat("#,##0");
  scope.getRange("E12:E13").setNumberFormat("#,##0");
  scope.getRange("E14").setNumberFormat("0.0%");
  scope.getRange("E15:E16").setNumberFormat("#,##0");
  scope.getRange("E17").setNumberFormat("0.0%");
}

workbook.recalculate();
const checks = [["Verified parties", poc ? 4 : 3],
                [detailName, poc ? 12 : 8], [reviewName, poc ? 13 : 5]];
if (poc && changes !== null && changes !== undefined) checks.push(["Changes", 10]);
if (poc && scenario) checks.push(["Scenario", 2]);
for (const [sheetName, width] of checks) {
  const checked = await workbook.inspect({kind: "table",
    range: `'${sheetName}'!A5:${String.fromCharCode(64 + width)}8`,
    include: "values,formulas", tableMaxRows: 4, tableMaxCols: width});
  console.log(checked.ndjson);
}
const errors = await workbook.inspect({kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: {useRegex: true, maxResults: 100}, summary: "formula error scan"});
console.log(errors.ndjson);
const output = path.join(outputDir, "suggestions_current.xlsx");
const result = await SpreadsheetFile.exportXlsx(workbook);
await result.save(output);
await fs.rm(output + ".inspect.ndjson", {force: true});
