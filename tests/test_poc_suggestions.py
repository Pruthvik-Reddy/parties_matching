import json
from pathlib import Path
from tempfile import TemporaryDirectory

from suggestions.poc import _review_candidates, build
from party_matching.domain import FinalDecision, MatchProposal, PartyRecord, load_config


def test_scoped_poc_rules_reassign_without_writing_state():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared = root / "prepared"
        prepared.mkdir()
        catalog = [
            {"partyId": "m", "partyName": "Microsoft Corporation"},
            {"partyId": "o", "partyName": "Other Party"},
        ]
        (prepared / "verified_parties.json").write_text(json.dumps(catalog), encoding="utf-8")
        (prepared / "jobs.json").write_text(json.dumps([{
            "accountId": "test", "confidenceCutoff": 0.0, "verifiedParties": catalog,
        }]), encoding="utf-8")
        records = [
            {"account_id": "test", "adm_party_id": str(i), "raw_name": name,
             "source_row": i, "eligible": True, "is_verified": False}
            for i, name in enumerate([
                "Microsoft Corp", "MICROSOFT", "Microsoft Azure Marketplace",
                "Microsfot Corporation", "Other Party", "Alphabet via Google",
                "Microsoft Corporation OBO Other Party",
            ], 1)
        ]
        (prepared / "adm_records.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        config = load_config("config.toml")
        config["paths"]["state_dir"] = str(root / "state")
        config["paths"]["artifacts_dir"] = str(root / "artifacts")
        config["execution"]["mode"] = "serial"
        config["retrieval"]["embedding_enabled"] = False
        initial, _, initial_summary = build(prepared, config, min_group_size=3)
        current, _, current_summary = build(
            prepared, config, min_group_size=3,
            added_names=["Microsoft Azure Marketplace"])
        assert initial_summary["counts"]["unverified_scored"] == 4
        assert current_summary["counts"]["unverified_scored"] == 4
        assert current_summary["counts"]["plain_unverified_scored"] == 3
        assert current_summary["counts"]["obo_unverified_scored"] == 1
        assert any(row["unverified_party"] == "Microsoft Azure Marketplace"
                   and row["verified_party"] == "Microsoft Azure Marketplace"
                   for row in current)
        assert not any(row["unverified_party"] == "Microsoft Azure Marketplace"
                       and row["verified_party"] == "Microsoft Corporation"
                       for row in current)
        assert initial_summary["scope"]["selected_anchor_groups"] == [
            {"anchor": "microsoft", "raw_unverified_rows": 4}]
        assert any(row["unverified_party"] == "Microsoft Corporation OBO Other Party"
                   and row["name_case"] == "OBO"
                   and row["matched_segment"] == "Microsoft Corporation"
                   and row["verified_party"] == "Microsoft Corporation"
                   for row in current)
        assert not (root / "state").exists()
        assert len(current) >= len(initial) - 1


def test_poc_suggestions_include_both_connectors_and_preserve_left_policy():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared = root / "prepared"
        prepared.mkdir()
        catalog = [
            {"partyId": "a", "partyName": "Alpha Corporation"},
            {"partyId": "b", "partyName": "Beta Corporation"},
        ]
        (prepared / "verified_parties.json").write_text(json.dumps(catalog), encoding="utf-8")
        (prepared / "jobs.json").write_text(json.dumps([{
            "accountId": "test", "confidenceCutoff": 0.0, "verifiedParties": catalog,
        }]), encoding="utf-8")
        names = ["Alpha Corporation OBO Beta Corporation",
                 "Beta Corporation VIA Alpha Corporation"]
        (prepared / "adm_records.jsonl").write_text(
            "\n".join(json.dumps({"account_id": "test", "adm_party_id": str(i),
                                   "raw_name": name, "source_row": i,
                                   "eligible": True, "is_verified": False})
                      for i, name in enumerate(names, 1)) + "\n", encoding="utf-8")
        config = load_config("config.toml")
        config["paths"]["state_dir"] = str(root / "state")
        config["paths"]["artifacts_dir"] = str(root / "artifacts")
        config["execution"]["mode"] = "serial"
        config["retrieval"]["embedding_enabled"] = False
        strong, review, summary = build(prepared, config)
        assert summary["counts"]["obo_unverified_scored"] == 1
        assert summary["counts"]["via_unverified_scored"] == 1
        assert len(strong) == 2
        assert not review
        assert {(row["name_case"], row["verified_party"], row["matched_segment"])
                for row in strong} == {
                    ("OBO", "Alpha Corporation", "Alpha Corporation"),
                    ("VIA", "Beta Corporation", "Beta Corporation"),
                }
        assert all(not row["connector_resolution"].endswith("RIGHT") for row in strong)
        assert not (root / "state").exists()


def test_connector_review_does_not_promote_a_different_segment():
    record = PartyRecord(account_id="test", adm_party_id="row", source_row=1,
                         raw_name="Alpha Corporation OBO Beta Corporation")
    decision = FinalDecision(account_id="test", adm_party_id="row", raw_name=record.raw_name,
                             decision="NO_MATCH", confidence=0.7,
                             reason="INSUFFICIENT_SUPPORT", connector_resolution="NONE")
    def proposal(position: int, name: str, score: float) -> MatchProposal:
        return MatchProposal(
            adm_party_id="row", mention_id=f"row:m{position}", mention_text=name,
            connector_before=None, connector_after=None,
            owner_party_id=name, owner_party_name=name,
            root_party_id=name, root_party_name=name,
            matched_name=name, candidate_type="official", candidate_source="official",
            candidate_confidence=None, char_tfidf_score=score, rules_score=score,
        )
    candidates = _review_candidates(record, decision,
                                    [proposal(0, "Alpha Corporation", 0.7),
                                     proposal(1, "Beta Corporation", 0.95)],
                                    {"matching": {}}, 0.6, 0.55, 2)
    assert len(candidates) == 1
    assert candidates[0]["verified_party"] == "Alpha Corporation"
    assert candidates[0]["matched_segment"] == "Alpha Corporation"
