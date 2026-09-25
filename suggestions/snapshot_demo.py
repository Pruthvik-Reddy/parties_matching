"""Two read-only verified-catalog snapshots using the enhanced POC rules.

This is a *simulated* earlier catalog, not an inferred corporate history. By
default, a few real verified-name groups are selected without using labels.
One representative per group stays verified initially; other names in those
groups are withheld. Both snapshots score the same unverified IDs. No graph,
mapping, event, or prepared input is written.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .engine import _GENERIC_ANCHORS, name_tokens
from .poc import _current_graph, _eligible_records, _write_csv, build
from .simple_workbook import export_snapshot
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

# Auto-selection is for choosing illustrative inputs, never for matching or
# asserting that two verified parties have the same owner.
_UNSAFE_AUTO_ANCHORS = _GENERIC_ANCHORS | {
    "american", "city", "county", "department", "federal", "state", "united",
}


def _auto_families(catalog: list[dict], eligible: list, max_families: int,
                   min_unverified: int) -> list[dict]:
    """Choose real multi-name groups with enough raw rows for a useful demo.

    The leading-token test is deliberately transparent and conservative. It
    does not use outcome labels, imply ownership, or fabricate added parties.
    """
    by_anchor: dict[str, list[dict]] = defaultdict(list)
    for party in catalog:
        tokens = name_tokens(str(party.get("partyName", "")))
        if tokens and len(tokens[0]) >= 4 and tokens[0] not in _UNSAFE_AUTO_ANCHORS:
            by_anchor[tokens[0]].append(party)
    by_anchor = {anchor: parties for anchor, parties in by_anchor.items()
                 if len({str(party["partyId"]) for party in parties}) >= 2}
    counts: Counter[str] = Counter()
    for item in eligible:
        present = {token for mention in parse_mentions(item)
                   for token in name_tokens(mention.text) if token in by_anchor}
        counts.update(present)
    ranked = sorted((anchor for anchor in by_anchor if counts[anchor] >= min_unverified),
                    key=lambda anchor: (-counts[anchor], anchor))[:max_families]
    families = []
    for anchor in ranked:
        parties = by_anchor[anchor]
        representative = min(parties, key=lambda party: (
            len(name_tokens(str(party["partyName"]))),
            len(str(party["partyName"])), str(party["partyName"]).casefold(),
            str(party["partyId"])))
        families.append({"anchor": anchor, "representative": representative,
                         "parties": parties, "eligible_unverified_with_token": counts[anchor]})
    return families


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


def build_snapshots(prepared: Path, config: dict, representative_name: str | None = None,
                    extra_names: list[str] | None = None,
                    all_unverified: bool = False,
                    extra_withheld_names: list[str] | None = None,
                    max_families: int = 3,
                    min_family_unverified: int = 3,
                    representative_list: list[dict] | None = None) -> tuple[tuple[list[dict], list[dict], dict],
                                                           tuple[list[dict], list[dict], dict],
                                                           list[dict], dict]:
    """Build comparable snapshots without using labels to choose a party."""
    catalog = read_json(prepared / "verified_parties.json", []) or []
    if representative_name and representative_list:
        raise ValueError("Use either --representative or --representatives-file, not both")
    if representative_name or representative_list:
        requested = (representative_list if representative_list is not None else
                     [{"name": representative_name}])
        if not requested:
            raise ValueError("The representative list is empty")
        if representative_list and (extra_names or extra_withheld_names):
            raise ValueError("--add-verified and --withhold-verified require a single --representative")
        families = []
        seen_anchors: set[str] = set()
        missing: list[str] = []
        for entry in requested:
            name = str(entry.get("name", "")).strip()
            representatives = [party for party in catalog if
                               str(party.get("partyName", "")).strip().casefold() == name.casefold()]
            if len(representatives) != 1:
                missing.append(name or "<blank>")
                continue
            representative = representatives[0]
            tokens = name_tokens(str(representative["partyName"]))
            if not tokens or len(tokens[0]) < 4:
                raise ValueError(f"Representative needs a distinctive leading name token: {name}")
            anchor = tokens[0]
            if anchor in seen_anchors:
                raise ValueError(f"Representatives share leading token '{anchor}'; choose one per group")
            seen_anchors.add(anchor)
            family = [party for party in catalog if
                      (parts := name_tokens(str(party.get("partyName", "")))) and parts[0] == anchor]
            families.append({"anchor": anchor, "representative": representative,
                             "parties": family, "eligible_unverified_with_token": None})
        if missing:
            raise ValueError("These representatives are not unique exact names in the prepared "
                             "verified catalog: " + "; ".join(missing))
    else:
        if extra_names or extra_withheld_names:
            raise ValueError("--add-verified and --withhold-verified require --representative")
        eligible, _ = _eligible_records(prepared)
        families = _auto_families(catalog, eligible, max_families, min_family_unverified)
        if not families:
            raise ValueError("No suitable multi-name verified groups in prepared data. "
                             "Use a fuller prepared catalog, lower --min-family-unverified, "
                             "or choose an existing --representative.")
    anchors = {family["anchor"] for family in families}
    representative_ids = {str(family["representative"]["partyId"]) for family in families}
    requested_withheld = {name.strip().casefold() for name in extra_withheld_names or [] if name.strip()}
    catalog_names = {str(party["partyName"]).strip().casefold() for party in catalog}
    if requested_withheld - catalog_names:
        raise ValueError("Each --withhold-verified must exactly name a current verified party")
    if any(str(party["partyName"]).strip().casefold() in requested_withheld
           for family in families for party in [family["representative"]]):
        raise ValueError("The representative cannot also be withheld")
    family_party_ids = {str(party["partyId"]) for family in families for party in family["parties"]}
    withheld = [party for party in catalog if str(party["partyId"]) not in representative_ids
                and (str(party["partyId"]) in family_party_ids or
                     str(party["partyName"]).strip().casefold() in requested_withheld)]
    initial_catalog = [party for party in catalog if party not in withheld]
    incoming = list(dict.fromkeys(name.strip() for name in extra_names or []
                                  if name.strip() and name.strip().casefold() not in catalog_names))
    if any(not name_tokens(name) or name_tokens(name)[0] not in anchors for name in incoming):
        raise ValueError("Each --add-verified name must share the representative's leading token")
    if not withheld and not incoming and representative_list is None:
        raise ValueError("No additional family verified parties exist; supply --add-verified to simulate one")

    if representative_name or representative_list:
        eligible, _ = _eligible_records(prepared)
    # Use the same broad, cheap family-name selection for both catalogs. Any
    # mention can contain the anchor, including an OBO/VIA segment. Count each
    # raw name once per group, without treating that count as known matches.
    family_input_counts: Counter[str] = Counter()
    selected_ids: set[str] = set()
    for item in eligible:
        present = {anchor for mention in parse_mentions(item)
                   for anchor in anchors.intersection(name_tokens(mention.text))}
        family_input_counts.update(present)
        if all_unverified or present:
            selected_ids.add(item.adm_party_id)
    for family in families:
        family["eligible_unverified_with_token"] = family_input_counts[family["anchor"]]
    if not selected_ids:
        raise ValueError("No eligible unverified names contain the selected family tokens; try --all-unverified")

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
    representative_roots = {expanded_graph.root_id(party_id) for party_id in representative_ids}
    same_root = [str(party["partyName"]) for party in withheld
                 if expanded_graph.root_id(str(party["partyId"])) in representative_roots]
    def coverage(result: tuple[list[dict], list[dict], dict]) -> dict:
        strong, review, _ = result
        review_ids = {row["unverified_party_id"] for row in review}
        covered = len(strong) + len(review_ids)
        return {"strong_names": len(strong), "review_only_names": len(review_ids),
                "no_suggestion_names": len(selected_ids) - covered,
                "suggestion_coverage": covered / len(selected_ids)}

    manifest = {
        "scenario": "simulated_earlier_verified_catalog",
        "selection_mode": ("attached_list" if representative_list is not None else
                           "explicit" if representative_name else "automatic"),
        "representative": " / ".join(str(family["representative"]["partyName"]) for family in families),
        "family_anchor": " / ".join(family["anchor"] for family in families),
        "families": [{"anchor": family["anchor"],
                      "representative": str(family["representative"]["partyName"]),
                      "verified_names": [str(party["partyName"]) for party in family["parties"]],
                      "eligible_unverified_with_token": family["eligible_unverified_with_token"]}
                     for family in families],
        "family_selection": "Same first meaningful name token plus explicitly withheld names; inspect list before presenting",
        "explicitly_withheld_names": sorted(requested_withheld),
        "withheld_initially": [{"partyId": str(party["partyId"]), "partyName": str(party["partyName"])}
                               for party in withheld],
        "added_only_in_expanded": incoming,
        "same_root_as_representative_in_expanded_graph": same_root,
        "input_selection": ("all eligible unverified names" if all_unverified else
                            "eligible names with a selected group token in any parsed segment"),
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
    parser.add_argument("--representative",
                        help="Optional exact verified-party name; otherwise select groups automatically")
    parser.add_argument("--representatives-file", type=Path,
                        help="JSON list of {name} verified parties to compare together")
    parser.add_argument("--max-families", type=int, default=3,
                        help="Automatic mode: most populous verified-name groups to compare (default: 3)")
    parser.add_argument("--min-family-unverified", type=int, default=3,
                        help="Automatic mode: minimum eligible unverified names containing the group token")
    parser.add_argument("--add-verified", action="append", default=[], metavar="NAME",
                        help="New name present only in expanded snapshot; repeat as needed")
    parser.add_argument("--withhold-verified", action="append", default=[], metavar="NAME",
                        help="Also withhold this exact current verified name initially; repeat as needed")
    parser.add_argument("--all-unverified", action="store_true",
                        help="Score all eligible names in both snapshots (slower); default selects family-token names")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xlsx", action="store_true")
    args = parser.parse_args()
    if args.max_families < 1 or args.min_family_unverified < 1:
        parser.error("--max-families and --min-family-unverified must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty; existing results are preserved")
    config = load_config(args.config)
    prepared = args.prepared_dir or Path(config.get("paths", {}).get("prepared_dir", "data/prepared"))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, str(config.get("execution", {}).get("native_threads", 10)))
    try:
        representative_list = None
        if args.representatives_file:
            representative_list = json.loads(args.representatives_file.read_text(encoding="utf-8"))
            if not isinstance(representative_list, list) or any(
                    not isinstance(entry, dict) for entry in representative_list):
                parser.error("--representatives-file must contain a JSON list of objects")
        initial, expanded, changes, manifest = build_snapshots(
            prepared, config, args.representative, args.add_verified, args.all_unverified,
            args.withhold_verified, args.max_families, args.min_family_unverified,
            representative_list)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_snapshot(args.output_dir / "initial", initial)
    _save_snapshot(args.output_dir / "expanded", expanded)
    _write_csv(args.output_dir / "changes.csv", changes, CHANGE_COLUMNS)
    (args.output_dir / "scenario.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.xlsx:
        export_snapshot(args.output_dir / "initial", initial)
        export_snapshot(args.output_dir / "expanded", expanded, changes=changes,
                        scenario=manifest)
    print(f"Scored the same {manifest['selected_unverified_names']} unverified names in both snapshots")
    print("Selected verified groups: " + ", ".join(
        f"{family['representative']} ({family['anchor']})" for family in manifest["families"]))
    print(f"Initially withheld {len(manifest['withheld_initially'])} verified family parties")
    print(f"Initial strong: {len(initial[0])}; expanded strong: {len(expanded[0])}; changes: {len(changes)}")
    if not changes:
        print("No suggestions changed for these inputs. The scenario contains no visible reassignment; "
              "choose a family with distinct matching names or add a real later verified party.")
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
