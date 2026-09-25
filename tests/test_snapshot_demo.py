import json
from pathlib import Path
from tempfile import TemporaryDirectory

from suggestions.snapshot_demo import build_snapshots
from party_matching.domain import load_config


def test_two_snapshots_use_same_inputs_and_separate_new_verified_party():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared = root / "prepared"
        prepared.mkdir()
        catalog = [
            {"partyId": "m", "partyName": "Microsoft Corporation"},
            {"partyId": "w", "partyName": "Microsoft Worldwide"},
            {"partyId": "o", "partyName": "Other Party"},
            {"partyId": "z", "partyName": "Azure Marketplace"},
        ]
        (prepared / "verified_parties.json").write_text(json.dumps(catalog), encoding="utf-8")
        (prepared / "jobs.json").write_text(json.dumps([{
            "accountId": "test", "confidenceCutoff": 0.0, "verifiedParties": catalog,
        }]), encoding="utf-8")
        names = ["Microsoft Corporation", "Microsoft Worldwide",
                 "Microsoft Worldwide - Contracts", "Other Party",
                 "Other Party OBO Microsoft Worldwide"]
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
        initial, expanded, changes, manifest = build_snapshots(
            prepared, config, "Microsoft Corporation",
            extra_withheld_names=["Azure Marketplace"])
        assert manifest["selected_unverified_names"] == 4
        assert [row["partyName"] for row in manifest["withheld_initially"]] == [
            "Microsoft Worldwide", "Azure Marketplace"]
        assert initial[2]["counts"]["unverified_scored"] == expanded[2]["counts"]["unverified_scored"] == 4
        assert not any(row["verified_party"] == "Microsoft Worldwide" for row in initial[0])
        assert any(row["unverified_party"] == "Microsoft Worldwide"
                   and row["verified_party"] == "Microsoft Worldwide" for row in expanded[0])
        assert any(row["unverified_party"] == "Microsoft Worldwide"
                   and row["change"] in {"MOVED", "NEW_STRONG"} for row in changes)
        assert not (root / "state").exists()


def test_snapshot_requires_existing_representative():
    with TemporaryDirectory() as temporary:
        prepared = Path(temporary)
        (prepared / "verified_parties.json").write_text("[]", encoding="utf-8")
        try:
            build_snapshots(prepared, load_config("config.toml"), "Microsoft Corporation")
        except ValueError as error:
            assert "exactly name one" in str(error)
        else:
            raise AssertionError("Expected a representative-selection error")
