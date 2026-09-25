"""Produce a complete current-state, verified-party-first suggestion snapshot.

This experiment is separate from the POC matcher. Suggestions are provisional;
the labels, graph, and accepted POC mappings are never read or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from .engine import (SuggestionIndex, VerifiedParty, change_status, name_tokens,
                     verified_name_relation)


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _verified_party(name: str, catalog: list[dict], *, allow_new: bool) -> VerifiedParty:
    matches = [item for item in catalog if str(item["partyName"]).casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Verified name is ambiguous in catalog: {name}")
    if matches:
        return VerifiedParty(str(matches[0]["partyId"]), str(matches[0]["partyName"]))
    if not allow_new:
        raise ValueError(f"Initial verified name is not in verified_parties.json: {name}")
    # A newly arriving party need not yet be in the prepared catalog.
    party_id = "incoming-" + hashlib.sha256(name.casefold().encode()).hexdigest()[:16]
    return VerifiedParty(party_id, name)


def _parties(names: list[str], catalog: list[dict], *, allow_new: bool) -> list[VerifiedParty]:
    parties = [_verified_party(name.strip(), catalog, allow_new=allow_new)
               for name in names if name.strip()]
    if len({party.party_id for party in parties}) != len(parties):
        raise ValueError("The same verified party was provided more than once")
    return parties


def build(prepared_dir: Path, initial_names: list[str], added_names: list[str],
          *, initial_all: bool = False) -> tuple[list[dict], list[dict], dict]:
    """Re-score every eligible plain row against the current verified list.

    The before snapshot is used only to count backend transitions. It is never
    written to the workbook: every exported suggestion reflects the *current*
    verified list after additions, including any implicit reassignment.
    """
    catalog = _read_json(prepared_dir / "verified_parties.json")
    if initial_all and initial_names:
        raise ValueError("Use --initial-all or --initial-verified, not both")
    if initial_all:
        initial = [VerifiedParty(str(item["partyId"]), str(item["partyName"]))
                   for item in catalog]
    else:
        initial = _parties(initial_names, catalog, allow_new=False)
    added = _parties(added_names, catalog, allow_new=True)
    if not initial:
        raise ValueError("Provide initial verified parties or --initial-all")
    if {party.party_id for party in initial} & {party.party_id for party in added}:
        raise ValueError("An added verified party already exists in the initial list")

    current = initial + added
    before_index = SuggestionIndex(initial) if added else None
    current_index = SuggestionIndex(current)
    suggestions: list[dict] = []
    ambiguous: list[dict] = []
    counts: Counter[str] = Counter()
    per_verified: Counter[str] = Counter()
    with (prepared_dir / "adm_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not record.get("eligible", True) or record.get("is_verified"):
                counts["excluded_ineligible_or_verified"] += 1
                continue
            raw_name = str(record.get("raw_name") or "")
            if not raw_name.strip():
                counts["excluded_blank_name"] += 1
                continue
            # Connector meaning is unresolved in this separate experiment. A
            # fragment of X OBO Y must not be treated as proof of either party.
            if re.search(r"\b(?:OBO|VIA)\b", raw_name, re.IGNORECASE):
                counts["excluded_connector"] += 1
                continue
            counts["eligible_plain_rows"] += 1
            result = current_index.suggest(raw_name)
            counts[f"current_{result.status.lower()}"] += 1
            if before_index is not None:
                previous = before_index.suggest(raw_name)
                counts[f"transition_{change_status(previous, result).lower()}"] += 1
            common = {
                "source_row": record.get("source_row", ""),
                "unverified_party_id": str(record.get("adm_party_id") or ""),
                "unverified_party": raw_name,
            }
            if result.status == "SUGGESTED":
                suggestions.append({
                    "verified_party_id": result.party_id,
                    "verified_party": result.party_name,
                    **common,
                    "evidence": result.evidence,
                    "status": "PROVISIONAL",
                })
                per_verified[result.party_id] += 1
            elif result.status == "AMBIGUOUS":
                ambiguous.append({
                    **common,
                    "possible_verified_parties": "; ".join(result.alternative_names),
                    "reason": result.evidence,
                })

    suggestions.sort(key=lambda row: (row["verified_party"].casefold(),
                                      row["unverified_party"].casefold(),
                                      str(row["unverified_party_id"])))
    for row in suggestions:
        row["verified_suggestion_count"] = per_verified[row["verified_party_id"]]
    ambiguous.sort(key=lambda row: (row["unverified_party"].casefold(),
                                    str(row["unverified_party_id"])))
    verified_summary = [
        {"verified_party_id": party.party_id, "verified_party": party.name,
         "suggestion_count": per_verified[party.party_id]}
        for party in current
    ]
    verified_summary.sort(key=lambda row: (-row["suggestion_count"],
                                           row["verified_party"].casefold()))
    assert len(suggestions) == sum(row["suggestion_count"] for row in verified_summary)
    assert counts["eligible_plain_rows"] == (
        counts["current_suggested"] + counts["current_ambiguous"] + counts["current_none"]
    )
    # Only compare plausible same-anchor verified names. Comparing every old
    # party with every arrival is unnecessary on a large prepared catalog.
    initial_by_anchor: dict[str, list[VerifiedParty]] = {}
    for party in initial:
        tokens = name_tokens(party.name)
        if tokens:
            initial_by_anchor.setdefault(tokens[0], []).append(party)
    relations: list[dict] = []
    for new in added:
        tokens = name_tokens(new.name)
        for old in initial_by_anchor.get(tokens[0], ()) if tokens else ():
            status, evidence = verified_name_relation(old, new)
            if status != "NO_NAME_RELATION":
                relations.append({"initial_name": old.name, "new_name": new.name,
                                  "status": status, "evidence": evidence})
    summary = {
        "workflow": "provisional_suggestions_current_snapshot",
        "initial_verified": [party.name for party in initial],
        "added_verified": [party.name for party in added],
        "counts": dict(counts),
        "verified_parties": verified_summary,
        "verified_relations_for_review": relations,
        "caveat": "Suggestions are not confirmed matches or proof of ownership. No labels were read.",
    }
    return suggestions, ambiguous, summary


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _export_xlsx(output_dir: Path, node: str, modules: Path,
                 suggestions: list[dict], ambiguous: list[dict], summary: dict,
                 changes: list[dict] | None = None, scenario: dict | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="suggestions_xlsx_", dir=output_dir) as temporary:
        working = Path(temporary)
        shutil.copyfile(Path(__file__).with_name("build_workbook.mjs"), working / "build_workbook.mjs")
        payload = working / "workbook_payload.json"
        payload.write_text(json.dumps({"suggestions": suggestions, "ambiguous": ambiguous,
                                       "summary": summary, "changes": changes,
                                       "scenario": scenario}, ensure_ascii=False), encoding="utf-8")
        module_link = working / "node_modules"
        try:
            module_link.symlink_to(modules.resolve(), target_is_directory=True)
        except OSError:
            import os
            if os.name != "nt":
                raise
            # Junction is scoped to this temporary working directory only.
            target = str(modules.resolve()).replace("'", "''")
            link = str(module_link).replace("'", "''")
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"New-Item -ItemType Junction -Path '{link}' -Target '{target}' | Out-Null"],
                           check=True)
        subprocess.run([node, str(working / "build_workbook.mjs"),
                        str(output_dir.resolve()), str(payload)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--initial-verified", action="append", default=[], metavar="NAME")
    parser.add_argument("--initial-all", action="store_true",
                        help="Use every currently prepared verified party as the initial list")
    parser.add_argument("--add-verified", action="append", default=[], metavar="NAME")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xlsx", action="store_true", help="Also export the complete current snapshot as XLSX")
    parser.add_argument("--artifact-modules", type=Path,
                        help="node_modules containing @oai/artifact-tool for XLSX")
    parser.add_argument("--node", help="Node.js executable for XLSX (default: node on PATH)")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be new or empty; existing results are preserved")
    node = args.node or shutil.which("node")
    modules = args.artifact_modules
    if args.xlsx and (not node or not modules or not (modules / "@oai" / "artifact-tool").exists()):
        parser.error("XLSX requires Node.js and --artifact-modules containing @oai/artifact-tool")
    try:
        suggestions, ambiguous, summary = build(args.prepared_dir, args.initial_verified,
                                                args.add_verified, initial_all=args.initial_all)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "suggestions_current.csv", suggestions,
               ("verified_party", "verified_suggestion_count", "unverified_party",
                "evidence", "source_row", "verified_party_id", "unverified_party_id", "status"))
    _write_csv(args.output_dir / "ambiguous_review.csv", ambiguous,
               ("unverified_party", "possible_verified_parties", "reason",
                "source_row", "unverified_party_id"))
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.xlsx:
        assert node is not None and modules is not None
        _export_xlsx(args.output_dir, node, modules, suggestions, ambiguous, summary)
    print(f"Wrote {len(suggestions)} current suggestions for {len(summary['verified_parties'])} verified parties")
    print(f"Wrote {len(ambiguous)} ambiguous rows for separate review")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
