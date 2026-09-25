"""Two read-only verified-catalog snapshots using the enhanced POC rules.

This is a *simulated* earlier catalog, not an inferred corporate history. The
representative stays verified; other current verified names sharing its first
meaningful token are temporarily withheld. Both snapshots score exactly the
same unverified IDs. No graph, mapping, event, or prepared input is written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .current import _export_xlsx
from .engine import name_tokens
from .poc import _current_graph, _eligible_records, _write_csv, build
from party_matching.domain import load_config, parse_mentions, read_json


STRONG_COLUMNS = (
    "verified_party", "verified_strong_count", "unverified_party", "rules_score",
    "match_method", "decision_tier", "name_case", "matched_segment",
    "connector_resolution", "source_row", "verified_party_id", "unverified_party_id", "status",
)
REVIEW_COLUMNS = (
    "verified_party", "verified_review_count", "unverified_party", "identity_score",
    "lexical_score", "decision_reason", "candidate_name", "name_case",
    "matched_segment", "connector_resolution", "source_row",
    "verified_party_id", "unverified_party_id", "status",
)
CHANGE_COLUMNS = (
    "change", "unverified_party", "name_case", "initial_status", "initial_verified_party",
    "initial_review_candidates", "expanded_status", "expanded_verified_party",
    "expanded_review_candidates", "unverified_party_id",
)


def _stage(strong: dict[str, dict], review: dict[str, list[dict]], adm_id: str) -> tuple[str, str, str]:
    accepted = strong.get(adm_id)
    if accepted:
        return "STRONG", accepted["verified_party"], accepted["verified_party_id"]
    candidates = sorted({(row["verified_party_id"], row["verified_party"])
                         for row in review.get(adm_id, [])})
    if candidates:
        return "REVIEW", " / ".join(name for _, name in candidates), " / ".join(
            party_id for party_id, _ in candidates)
    return "NONE", "", ""


def _transition(before: tuple[str, str, str], after: tuple[str, str, str]) -> str:
    if before[0] == "STRONG" and after[0] == "STRONG":
        return "MOVED" if before[2] != after[2] else "UNCHANGED"
    if before[0] == "STRONG":
        return "LOST_STRONG"
    if after[0] == "STRONG":
        return "NEW_STRONG"
    return "REVIEW_CHANGED" if before != after else "UNCHANGED"


def build_snapshots(prepared: Path, config: dict, representative_name: str,
                    extra_names: list[str] | None = None,
                    all_unverified: bool = False,
                    extra_withheld_names: list[str] | None = None) -> tuple[tuple[list[dict], list[dict], dict],
                                                           tuple[list[dict], list[dict], dict],
                                                           list[dict], dict]:
    """Build comparable snapshots without using labels to choose a party."""
    catalog = read_json(prepared / "verified_parties.json", []) or []
    representatives = [party for party in catalog if
                       str(party.get("partyName", "")).strip().casefold() ==
                       representative_name.strip().casefold()]
    if len(representatives) != 1:
        raise ValueError("--representative must exactly name one current verified party")
    representative = representatives[0]
    representative_id = str(representative["partyId"])
    tokens = name_tokens(str(representative["partyName"]))
    if not tokens or len(tokens[0]) < 4:
        raise ValueError("Representative needs a distinctive leading name token")
    anchor = tokens[0]
    family = [party for party in catalog if
              (parts := name_tokens(str(party.get("partyName", "")))) and parts[0] == anchor]
    requested_withheld = {name.strip().casefold() for name in extra_withheld_names or [] if name.strip()}
    catalog_names = {str(party["partyName"]).strip().casefold() for party in catalog}
    if requested_withheld - catalog_names:
        raise ValueError("Each --withhold-verified must exactly name a current verified party")
    if str(representative["partyName"]).strip().casefold() in requested_withheld:
        raise ValueError("The representative cannot also be withheld")
    withheld = [party for party in catalog if str(party["partyId"]) != representative_id
                and (party in family or str(party["partyName"]).strip().casefold() in requested_withheld)]
    initial_catalog = [party for party in catalog if party not in withheld]
    incoming = list(dict.fromkeys(name.strip() for name in extra_names or []
                                  if name.strip() and name.strip().casefold() not in catalog_names))
    if any(not name_tokens(name) or name_tokens(name)[0] != anchor for name in incoming):
        raise ValueError("Each --add-verified name must share the representative's leading token")
    if not withheld and not incoming:
        raise ValueError("No additional family verified parties exist; supply --add-verified to simulate one")

    eligible, _ = _eligible_records(prepared)
    if all_unverified:
        selected_ids = {item.adm_party_id for item in eligible}
    else:
        # Use the same broad, cheap family-name selection for both catalogs.
        # Any mention can contain the anchor, including an OBO/VIA segment.
        selected_ids = {
            item.adm_party_id for item in eligible
            if any(anchor in name_tokens(mention.text) for mention in parse_mentions(item))
        }
    if not selected_ids:
        raise ValueError("No eligible unverified names contain the family token; try --all-unverified")

    options = dict(selected_record_ids=selected_ids, review_score_floor=0.45,
                   review_lexical_floor=0.45, review_max_candidates=3,
                   review_within_top=0.15)
    initial = build(prepared, config, catalog_override=initial_catalog, **options)
    expanded = build(prepared, config, catalog_override=catalog,
                     added_names=incoming, **options)
    initial[2]["snapshot_stage"] = "Initial simulated catalog"
    expanded[2]["snapshot_stage"] = "Expanded current catalog"

    names = {item.adm_party_id: item for item in eligible if item.adm_party_id in selected_ids}
    before_strong, before_review, _ = initial
    after_strong, after_review, _ = expanded
    initial_strong_by_id = {row["unverified_party_id"]: row for row in before_strong}
    expanded_strong_by_id = {row["unverified_party_id"]: row for row in after_strong}
    initial_review_by_id: dict[str, list[dict]] = defaultdict(list)
    expanded_review_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in before_review:
        initial_review_by_id[row["unverified_party_id"]].append(row)
    for row in after_review:
        expanded_review_by_id[row["unverified_party_id"]].append(row)
    case_by_id = {row["unverified_party_id"]: row["name_case"]
                  for row in [*before_strong, *after_strong, *before_review, *after_review]}
    changes: list[dict] = []
    movement_counts: Counter[str] = Counter()
    for adm_id, record in names.items():
        before = _stage(initial_strong_by_id, initial_review_by_id, adm_id)
        after = _stage(expanded_strong_by_id, expanded_review_by_id, adm_id)
        movement = _transition(before, after)
        movement_counts[movement] += 1
        if movement == "UNCHANGED":
            continue
        changes.append({
            "change": movement, "unverified_party": record.raw_name,
            "name_case": case_by_id.get(adm_id, ""),
            "initial_status": before[0],
            "initial_verified_party": before[1] if before[0] == "STRONG" else "",
            "initial_review_candidates": before[1] if before[0] == "REVIEW" else "",
            "expanded_status": after[0],
            "expanded_verified_party": after[1] if after[0] == "STRONG" else "",
            "expanded_review_candidates": after[1] if after[0] == "REVIEW" else "",
            "unverified_party_id": adm_id,
        })
    changes.sort(key=lambda row: (row["change"], row["unverified_party"].casefold(),
                                  row["unverified_party_id"]))

    jobs = read_json(prepared / "jobs.json", []) or []
    account_id = str(jobs[0]["accountId"]) if jobs else ""
    expanded_graph = _current_graph(prepared, config, account_id, incoming, catalog)
    representative_root = expanded_graph.root_id(representative_id)
    same_root = [str(party["partyName"]) for party in withheld
                 if expanded_graph.root_id(str(party["partyId"])) == representative_root]
    def coverage(result: tuple[list[dict], list[dict], dict]) -> dict:
        strong, review, _ = result
        review_ids = {row["unverified_party_id"] for row in review}
        covered = len(strong) + len(review_ids)
        return {"strong_names": len(strong), "review_only_names": len(review_ids),
                "no_suggestion_names": len(selected_ids) - covered,
                "suggestion_coverage": covered / len(selected_ids)}

    manifest = {
        "scenario": "simulated_earlier_verified_catalog",
        "representative": str(representative["partyName"]),
        "family_anchor": anchor,
        "family_selection": "Same first meaningful name token plus explicitly withheld names; inspect list before presenting",
        "explicitly_withheld_names": sorted(requested_withheld),
        "withheld_initially": [{"partyId": str(party["partyId"]), "partyName": str(party["partyName"])}
                               for party in withheld],
        "added_only_in_expanded": incoming,
        "same_root_as_representative_in_expanded_graph": same_root,
        "input_selection": ("all eligible unverified names" if all_unverified else
                            "eligible names with family token in any parsed segment"),
        "selected_unverified_names": len(selected_ids),
        "initial_coverage": coverage(initial),
        "expanded_coverage": coverage(expanded),
        "movement_counts": dict(movement_counts),
        "review_policy": "Broader, unvalidated review candidates; not accepted matches",
        "warning": "This is not historical onboarding data or a full-dataset recall measurement",
    }
    return initial, expanded, changes, manifest


def _save_snapshot(folder: Path, result: tuple[list[dict], list[dict], dict]) -> None:
    strong, review, summary = result
    folder.mkdir(parents=True, exist_ok=True)
    _write_csv(folder / "strong_suggestions.csv", strong, STRONG_COLUMNS)
    _write_csv(folder / "review_candidates.csv", review, REVIEW_COLUMNS)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--representative", required=True,
                        help="Exact current verified-party name retained as the family's initial representative")
    parser.add_argument("--add-verified", action="append", default=[], metavar="NAME",
                        help="New name present only in expanded snapshot; repeat as needed")
    parser.add_argument("--withhold-verified", action="append", default=[], metavar="NAME",
                        help="Also withhold this exact current verified name initially; repeat as needed")
    parser.add_argument("--all-unverified", action="store_true",
                        help="Score all eligible names in both snapshots (slower); default selects family-token names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xlsx", action="store_true")
    parser.add_argument("--node", help="Node.js executable for XLSX")
    parser.add_argument("--artifact-modules", type=Path,
                        help="node_modules containing @oai/artifact-tool for XLSX")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; existing results are preserved")
    config = load_config(args.config)
    prepared = args.prepared_dir or Path(config.get("paths", {}).get("prepared_dir", "data/prepared"))
    bundled_node = (Path.home() / ".cache" / "codex-runtimes" /
                    "codex-primary-runtime" / "dependencies" / "node")
    bundled_exe = bundled_node / "bin" / ("node.exe" if os.name == "nt" else "node")
    node = args.node or (str(bundled_exe) if bundled_exe.exists() else shutil.which("node"))
    modules = args.artifact_modules or (bundled_node / "node_modules")
    if args.xlsx and (not node or not (modules / "@oai" / "artifact-tool").exists()):
        parser.error("XLSX requires Node.js and --artifact-modules containing @oai/artifact-tool")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(config.get("execution", {}).get("native_threads", 10)))
    try:
        initial, expanded, changes, manifest = build_snapshots(
            prepared, config, args.representative, args.add_verified, args.all_unverified,
            args.withhold_verified)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_snapshot(args.output_dir / "initial", initial)
    _save_snapshot(args.output_dir / "expanded", expanded)
    _write_csv(args.output_dir / "changes.csv", changes, CHANGE_COLUMNS)
    (args.output_dir / "scenario.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.xlsx:
        _export_xlsx(args.output_dir / "initial", node, modules, *initial)
        _export_xlsx(args.output_dir / "expanded", node, modules, *expanded, changes=changes,
                     scenario=manifest)
    print(f"Scored the same {manifest['selected_unverified_names']} unverified names in both snapshots")
    print(f"Initially withheld {len(manifest['withheld_initially'])} verified family parties")
    print(f"Initial strong: {len(initial[0])}; expanded strong: {len(expanded[0])}; changes: {len(changes)}")
    if not changes:
        print("No suggestions changed for these inputs. The scenario contains no visible reassignment; "
              "choose a family with distinct matching names or add a real later verified party.")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
