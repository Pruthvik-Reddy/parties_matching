import json
from pathlib import Path
from tempfile import TemporaryDirectory

from suggestions.current import build
from suggestions.engine import VerifiedParty, change_status, suggest, verified_name_relation


def test_new_specific_verified_moves_only_provisional_suggestion():
    initial = [VerifiedParty("a", "Abbott Laboratories")]
    later = initial + [VerifiedParty("m", "Abbott Molecular")]
    before = suggest("Abbott Molecular", initial)
    after = suggest("Abbott Molecular", later)
    assert before.party_id == "a"
    assert "relationship unverified" in before.evidence
    assert after.party_id == "m"
    assert change_status(before, after) == "MOVED_SUGGESTION"


def test_new_verified_does_not_force_related_name_to_one_root():
    verified = [VerifiedParty("a", "Abbott Laboratories"),
                VerifiedParty("m", "Abbott Molecular")]
    answer = suggest("Abbott Nutrition", verified)
    assert answer.status == "AMBIGUOUS"
    assert not answer.party_id


def test_legal_form_and_generic_anchor():
    verified = [VerifiedParty("a", "Abbott Laboratories")]
    assert suggest("Abbott Labs Ltd", verified).party_id == "a"
    assert suggest("Bank Treasury Department", [VerifiedParty("b", "Bank N.A.")]).status == "NONE"


def test_legal_form_breaks_tie_but_bare_name_stays_ambiguous():
    parties = [VerifiedParty("i", "Tock, Inc."), VerifiedParty("l", "Tock, LLC")]
    assert suggest("Tock Inc", parties).party_id == "i"
    assert suggest("Tock", parties).status == "AMBIGUOUS"


def test_complete_current_grouping_and_reassignment():
    with TemporaryDirectory() as temporary:
        prepared = Path(temporary)
        (prepared / "verified_parties.json").write_text(
            json.dumps([{"partyId": "a", "partyName": "Abbott Laboratories"}]),
            encoding="utf-8",
        )
        names = ["Abbott Laboratories", "Abbott Labs Ltd", "Abbott Molecular",
                 "Abbott Nutrition", "Abbott Molecular Office", "Other Name",
                 "Abbott OBO Other Name"]
        records = [
            {"adm_party_id": str(index), "raw_name": name, "source_row": index,
             "eligible": True, "is_verified": False}
            for index, name in enumerate(names, 1)
        ]
        (prepared / "adm_records.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        rows, review, summary = build(prepared, ["Abbott Laboratories"],
                                      ["Abbott Molecular"])
    assert len(rows) == 4  # Every current proposal, not a sampled subset.
    assert len(review) == 1
    by_name = {row["unverified_party"]: row for row in rows}
    assert by_name["Abbott Molecular"]["verified_party"] == "Abbott Molecular"
    assert by_name["Abbott Molecular Office"]["verified_party"] == "Abbott Molecular"
    assert by_name["Abbott Labs Ltd"]["verified_party"] == "Abbott Laboratories"
    assert by_name["Abbott Molecular"]["verified_suggestion_count"] == 2
    assert summary["counts"]["transition_moved_suggestion"] == 2
    assert summary["counts"]["excluded_connector"] == 1
    assert sum(row["suggestion_count"] for row in summary["verified_parties"]) == len(rows)


def test_export_logic_has_no_fifteen_row_cap():
    with TemporaryDirectory() as temporary:
        prepared = Path(temporary)
        (prepared / "verified_parties.json").write_text(
            json.dumps([{"partyId": "a", "partyName": "AcmeTech Corp"}]), encoding="utf-8")
        records = [
            {"adm_party_id": str(index), "raw_name": f"AcmeTech Corp Location {index}",
             "eligible": True, "is_verified": False}
            for index in range(30)
        ]
        (prepared / "adm_records.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        rows, review, summary = build(prepared, ["AcmeTech Corp"], [])
    assert len(rows) == 30
    assert not review
    assert summary["verified_parties"][0]["suggestion_count"] == 30


def test_new_verified_relation_is_review_only():
    old = VerifiedParty("a", "Abbott Laboratories")
    assert verified_name_relation(old, VerifiedParty("b", "Abbott Laboratories Ltd"))[0] == "POSSIBLE_DUPLICATE_NAME"
    assert verified_name_relation(old, VerifiedParty("m", "Abbott Molecular"))[0] == "POSSIBLE_NAME_FAMILY"
