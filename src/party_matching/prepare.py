from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .domain import content_hash, normalize_name, stable_id, write_json, write_jsonl


RAW_CANDIDATES = ("Raw Name (original)", "Raw Name", "raw_name")
CANONICAL_CANDIDATES = ("Resolved (Canonical) Name", "Resolved Name", "canonical_name")
CATEGORY_CANDIDATES = ("Category", "category")
PARENT_ID_CANDIDATES = ("Parent ID", "Parent", "parent_id")
PARENT_VERIFIED_CANDIDATES = ("Parent Verified", "Is Parent Verified", "parent_is_verified")
RECORD_VERIFIED_CANDIDATES = ("Verified", "Is Verified", "is_verified")


def prepare_workbook(
    workbook: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = Path(workbook)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = config.get("workbook", {}).get("detail_sheet", "Detail")
    if source.suffix.casefold() == ".csv":
        frame = pd.read_csv(source)
    else:
        frame = pd.read_excel(source, sheet_name=sheet, engine="openpyxl")
    raw_col = _find_column(frame, config.get("workbook", {}).get("raw_name_column"), RAW_CANDIDATES)
    canonical_col = _find_column(frame, config.get("workbook", {}).get("canonical_name_column"), CANONICAL_CANDIDATES)
    category_col = _find_column(frame, config.get("workbook", {}).get("category_column"), CATEGORY_CANDIDATES, required=False)
    parent_id_col = _find_column(frame, config.get("workbook", {}).get("parent_id_column"), PARENT_ID_CANDIDATES, required=False)
    parent_verified_col = _find_column(frame, config.get("workbook", {}).get("parent_verified_column"), PARENT_VERIFIED_CANDIDATES, required=False)
    record_verified_col = _find_column(frame, config.get("workbook", {}).get("record_verified_column"), RECORD_VERIFIED_CANDIDATES, required=False)
    account_id = str(config.get("account", {}).get("id", "poc-account"))

    clean = frame.where(pd.notna(frame), None)
    canonical_by_key: dict[str, str] = {}
    for value in clean[canonical_col].tolist():
        name = str(value or "").strip()
        key = normalize_name(name)
        if key and key != "unknown":
            canonical_by_key.setdefault(key, name)
    canonical_values = sorted(canonical_by_key.values(), key=str.casefold)
    parties = [{
        "partyId": stable_id("verified", account_id, normalize_name(name)),
        "partyName": name,
    } for name in canonical_values]
    party_ids = {normalize_name(item["partyName"]): item["partyId"] for item in parties}
    verified_references = set(party_ids) | {item["partyId"] for item in parties}

    adm_rows, label_rows, source_rows = [], [], []
    split_counts: Counter[str] = Counter()
    for offset, row in enumerate(clean.to_dict(orient="records"), start=2):
        raw_name = str(row.get(raw_col) or "").strip()
        canonical = str(row.get(canonical_col) or "").strip()
        category = str(row.get(category_col) or "") if category_col else ""
        adm_id = stable_id("adm", account_id, offset, raw_name)
        parent_id = str(row.get(parent_id_col) or "").strip() if parent_id_col else ""
        parent_is_verified = (
            _as_bool(row.get(parent_verified_col))
            if parent_verified_col
            else bool(parent_id and (parent_id in verified_references or normalize_name(parent_id) in verified_references))
        )
        record_is_verified = _as_bool(row.get(record_verified_col)) if record_verified_col else False
        eligible = not parent_id or not parent_is_verified
        has_known_label = bool(canonical and canonical.casefold() != "unknown")
        scorable = has_known_label and eligible
        if not eligible:
            split = "INELIGIBLE"
        elif has_known_label:
            split = _split_for(canonical, raw_name)
        else:
            split = "UNKNOWN"
        split_counts[split] += 1
        adm_rows.append({
            "account_id": account_id,
            "adm_party_id": adm_id,
            "raw_name": raw_name,
            "source_row": offset,
            "eligible": eligible,
            "parent_id": parent_id or None,
            "parent_is_verified": parent_is_verified,
            "is_verified": record_is_verified,
        })
        label_rows.append({
            "adm_party_id": adm_id,
            "expected_canonical_name": canonical or None,
            "expected_party_id": party_ids.get(normalize_name(canonical)),
            "category": category or None,
            "split": split,
            "scorable": scorable,
            "exclusion_reason": (
                None if scorable else
                "INELIGIBLE_PARENT_VERIFIED" if not eligible else
                "UNKNOWN_OR_EMPTY_LABEL"
            ),
        })
        source_rows.append({
            "adm_party_id": adm_id,
            "source_row": offset,
            "source": row,
        })

    write_json(out / "verified_parties.json", parties)
    write_jsonl(out / "adm_records.jsonl", adm_rows)
    write_jsonl(out / "labels.jsonl", label_rows)
    write_jsonl(out / "source_rows.jsonl", source_rows)
    jobs = []
    for index in range(0, len(parties), 100):
        chunk = parties[index:index + 100]
        jobs.append({
            "accountId": account_id,
            "confidenceCutoff": float(config.get("decision", {}).get("request_confidence_cutoff", 0.0)),
            "jobId": stable_id("job", account_id, index // 100),
            "shardId": f"{index // 100 + 1}.us",
            "verifiedParties": chunk,
        })
    write_json(out / "jobs.json", jobs)
    source_hash = _frame_hash(frame)
    manifest = {
        "source": str(source.resolve()),
        "account_id": account_id,
        "rows": len(frame),
        "verified_parties": len(parties),
        "source_columns": list(frame.columns),
        "raw_name_column": raw_col,
        "canonical_name_column": canonical_col,
        "category_column": category_col,
        "parent_id_column": parent_id_col,
        "parent_verified_column": parent_verified_col,
        "record_verified_column": record_verified_col,
        "split_counts": dict(split_counts),
        "source_hash": source_hash,
        "data_version": content_hash({
            "source_hash": source_hash,
            "parties": parties,
            "columns": list(frame.columns),
        }),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def create_sample_workbook(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        "Microsoft Corporation": [
            "Microsoft Corporation", "Microsoft Corp", "MICROSOFT", "Microsfot Corporation",
            "Microsoft Azure Marketplace", "MSFT", "GitHub OBO Microsoft Corporation",
        ],
        "LinkedIn Corporation": ["LinkedIn Corporation", "Linked In Corp", "LinkedIn", "LINKEDIN INC"],
        "Google LLC": ["Google LLC", "Google", "GOOGLE INC", "Alphabet via Google"],
        "3M Company": ["3M Company", "3 M Company", "3M", "3M United Kingdom PLC"],
        "VA Financial Services LLC": ["VA Financial Services LLC", "VA Financial Services", "V A Financial Services"],
        "ER Financial Services LLC": ["ER Financial Services LLC", "ER Financial Services", "E R Financial Services"],
        "Local Engines Inc": ["Local Engines Inc", "Local Engines", "Local Engines United Kingdom"],
        "Arizona Department of Gaming": [
            "Arizona Department of Gaming", "AZ Department of Gaming",
            "Carahsoft OBO Arizona Department of Gaming",
        ],
        "International Business Machines Corporation": [
            "International Business Machines Corporation", "IBM", "I.B.M.", "Intl Business Machines",
        ],
        "Amazon Web Services Inc": ["Amazon Web Services Inc", "Amazon Web Services", "AWS", "A W S"],
    }
    rows = []
    for canonical, aliases in groups.items():
        for raw in aliases:
            rows.append({
                "Raw Name (original)": raw,
                "Resolved (Canonical) Name": canonical,
                "Category": "Corporate",
                "Aliases in Group": len(aliases),
                "Parent ID": "",
                "Parent Verified": False,
                "Is Verified": raw == canonical,
            })
    rows.extend([
        {"Raw Name (original)": "Existing Microsoft Child", "Resolved (Canonical) Name": "Microsoft Corporation", "Category": "Corporate", "Aliases in Group": 1, "Parent ID": "verified-parent", "Parent Verified": True, "Is Verified": False},
        {"Raw Name (original)": "FES", "Resolved (Canonical) Name": "UNKNOWN", "Category": "UNKNOWN", "Aliases in Group": 1, "Parent ID": "", "Parent Verified": False, "Is Verified": False},
        {"Raw Name (original)": "", "Resolved (Canonical) Name": "UNKNOWN", "Category": "UNKNOWN", "Aliases in Group": 1, "Parent ID": "", "Parent Verified": False, "Is Verified": False},
        {"Raw Name (original)": "Unrelated Neighborhood Club", "Resolved (Canonical) Name": "UNKNOWN", "Category": "UNKNOWN", "Aliases in Group": 1, "Parent ID": "", "Parent Verified": False, "Is Verified": False},
    ])
    detail = pd.DataFrame(rows)
    summary = (detail[detail["Resolved (Canonical) Name"] != "UNKNOWN"]
               .groupby("Resolved (Canonical) Name", as_index=False)
               .agg(**{"# Raw Names": ("Raw Name (original)", "count"),
                       "Raw Aliases (all variants)": ("Raw Name (original)", lambda values: " | ".join(values))}))
    with pd.ExcelWriter(target, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        detail.to_excel(writer, sheet_name="Detail", index=False)
    return target


def _find_column(frame: pd.DataFrame, configured: str | None, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {normalize_name(column): column for column in frame.columns}
    for candidate in (configured, *candidates):
        if candidate and normalize_name(candidate) in lookup:
            return lookup[normalize_name(candidate)]
    if required:
        raise ValueError(f"Missing workbook column. Expected one of: {', '.join(candidates)}")
    return None


def _split_for(canonical: str, raw_name: str) -> str:
    family_bucket = int(content_hash({"canonical": normalize_name(canonical)})[:8], 16) % 100
    if family_bucket < 15:
        return "TEST_UNSEEN"
    template = normalize_name(raw_name)
    template_bucket = int(content_hash({"canonical": normalize_name(canonical), "template": template})[:8], 16) % 100
    if template_bucket < 15:
        return "TEST_KNOWN"
    if template_bucket < 30:
        return "CALIBRATION"
    return "TRAIN"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) and not (isinstance(value, float) and math.isnan(value))
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "verified"}


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash values as well as shape so stale model artifacts can be rejected."""
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:16]
