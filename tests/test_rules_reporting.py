import unittest

from party_matching.reporting import _metrics, _rules_demo_rows


class RulesDemoReportTests(unittest.TestCase):
    def test_rules_projection_preserves_ml_and_counts_no_match_in_recall(self):
        base = {
            "Dataset Split": "TEST_UNSEEN", "Held-Out": True,
            "Scorable": True, "Evaluation Status": "LABELED_HELD_OUT",
            "Category": "Corporate", "Has OBO": False, "Has VIA": False,
            "Expected Canonical Name": "Acme Corp", "Expected Global Parent Name": "Acme Corp",
            "Expected Verified ID": "acme", "Decision": "NO_MATCH", "Correct": False,
            "Reason": "INSUFFICIENT_SUPPORT", "Confidence": 0.2,
            "Predicted Verified ID": None, "Predicted Verified Name": None,
            "Expected Target Retrieved": False, "Expected Target Rank": None,
            "Error Bucket": "RETRIEVAL_MISS", "Case Types": "GENERAL",
            "Connector Resolution": None, "Rules-only Verified ID": None,
            "Rules-only Decision Tier": None, "Prior Rules Verified ID": None,
            "Prior Rules Decision Tier": None,
        }
        first = dict(base, **{"ADM Party ID": "one"})
        second = dict(base, **{"ADM Party ID": "two"})
        rows = [
            {"source": {"Raw": "Acme"}, "prediction": first},
            {"source": {"Raw": "Unresolved"}, "prediction": second},
        ]
        rules = {
            "one": {"decision": "MATCH", "verified_party_id": "acme",
                    "verified_party_name": "Acme Corp", "confidence": 0.95,
                    "reason": "MATCH", "retrieved_root_ids": ["acme"]},
            "two": {"decision": "NO_MATCH", "confidence": 0.35,
                    "reason": "INSUFFICIENT_SUPPORT", "retrieved_root_ids": ["acme"]},
        }

        projected = _rules_demo_rows(rows, rules, {"raw_name_column": "Raw"})
        result = _metrics(projected, {})["overall"]

        self.assertEqual(rows[0]["prediction"]["Decision"], "NO_MATCH")
        self.assertEqual(projected[0]["prediction"]["Predicted Verified Name"], "Acme Corp")
        self.assertEqual(projected[1]["prediction"]["Decision"], "NO_MATCH")
        self.assertEqual(result["scorable"], 2)
        self.assertEqual(result["correct_matches"], 1)
        self.assertEqual(result["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
