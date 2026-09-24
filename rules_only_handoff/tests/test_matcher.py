import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from party_matching_rules import match_parties


ROOT = Path(__file__).resolve().parents[1]


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.verified = [
            {"verified_id": "carahsoft", "verified_name": "Carahsoft Technology Corp."},
            {"verified_id": "afrl", "verified_name": "Air Force Research Laboratory"},
            {"verified_id": "root", "verified_name": "Acme Holdings Inc."},
            {"verified_id": "child", "verified_name": "Acme Services LLC", "parent_id": "root"},
        ]

    def test_short_name_and_parent_root(self):
        result = match_parties(self.verified, [
            {"unverified_id": "u1", "unverified_name": "Carahsoft"},
            {"unverified_id": "u2", "unverified_name": "Acme Services LLC"},
        ])
        self.assertEqual((result[0]["decision"], result[0]["verified_id"]), ("MATCH", "carahsoft"))
        self.assertEqual((result[1]["decision"], result[1]["verified_id"]), ("MATCH", "root"))

    def test_obo_and_via_prefer_left_exact_part(self):
        result = match_parties(self.verified, [
            {"unverified_id": "u1", "unverified_name": "Carahsoft Technology Corp. OBO Air Force Research Laboratory"},
            {"unverified_id": "u2", "unverified_name": "Carahsoft Technology Corp. VIA Air Force Research Laboratory"},
        ])
        self.assertEqual([row["verified_id"] for row in result], ["carahsoft", "carahsoft"])

    def test_obo_and_via_recover_unique_short_left_part(self):
        result = match_parties(self.verified, [
            {"unverified_id": "u1", "unverified_name": "Carahsoft OBO Air Force Research Laboratory"},
            {"unverified_id": "u2", "unverified_name": "Carahsoft VIA Air Force Research Laboratory"},
        ])
        self.assertEqual([row["verified_id"] for row in result], ["carahsoft", "carahsoft"])
        self.assertEqual([row["decision_tier"] for row in result],
                         ["RULES_ROOT_UNIQUE_CONNECTOR_SHORT_NAME"] * 2)

    def test_enhanced_safe_name_views(self):
        result = match_parties(self.verified, [
            {"unverified_id": "u1", "unverified_name": "Carahsoft Technology - Partner"},
            {"unverified_id": "u2", "unverified_name": "Carahsoft Technologies"},
        ])
        self.assertEqual([row["verified_id"] for row in result], ["carahsoft", "carahsoft"])
        self.assertEqual(result[0]["decision_tier"], "RULES_SAFE_VIEW_FIRST_SEGMENT")
        self.assertEqual(result[1]["decision_tier"], "RULES_SAFE_VIEW_PLURAL_IES")

    def test_rejects_invalid_parent(self):
        with self.assertRaisesRegex(ValueError, "Unknown parent_id"):
            match_parties([{"verified_id": "v1", "verified_name": "A", "parent_id": "missing"}], [])

    def test_csv_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            subprocess.run([
                sys.executable, str(ROOT / "match_csv.py"),
                "--verified", str(ROOT / "examples" / "verified.csv"),
                "--unverified", str(ROOT / "examples" / "unverified.csv"),
                "--output", str(output),
            ], check=True, capture_output=True, text=True)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["verified_id"], "carahsoft")
            self.assertEqual(rows[2]["verified_id"], "acme_root")
            self.assertEqual(rows[3]["decision"], "NO_MATCH")


if __name__ == "__main__":
    unittest.main()
