import unittest

from party_matching.domain import MatchProposal, PartyRecord, parse_mentions
from party_matching.graph import VerifiedGraph
from party_matching.matching import decide_records


class IdentityScorer:
    def __init__(self, plain_threshold):
        self.system_threshold = 0.80
        self.plain_threshold = plain_threshold

    def score_target_vectors(self, vectors, batch_size=50_000):
        return [vector[0] for vector in vectors]


class NoReranker:
    def score(self, proposals):
        return None


class ConnectorPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.graph = VerifiedGraph("account", "unused-graph.json", load_existing=False)
        self.graph.add_parties([
            {"partyId": "left", "partyName": "Alpha"},
            {"partyId": "right", "partyName": "Beta"},
        ])
        self.config = {"matching": {
            "connector_policy": "positional", "cross_encoder_mode": "off",
            "preferred_near_cutoff_band": 0.03,
        }}

    def decide(self, connector, left_score, right_score, plain_threshold):
        record = PartyRecord("account", "raw-1", f"Alpha {connector} Beta", 1)
        proposals = []
        for mention, root_id, score in zip(parse_mentions(record), ("left", "right"), (left_score, right_score)):
            proposals.append(MatchProposal(
                adm_party_id=record.adm_party_id, mention_id=mention.mention_id,
                mention_text=mention.text, connector_before=mention.connector_before,
                connector_after=mention.connector_after, owner_party_id=root_id,
                owner_party_name=mention.text, root_party_id=root_id,
                root_party_name=mention.text, matched_name=mention.text,
                candidate_type="official", candidate_source="official",
                candidate_confidence=None, feature_score=score,
            ))
        return decide_records(
            [record], {record.adm_party_id: proposals}, self.graph,
            IdentityScorer(plain_threshold), NoReranker(), self.config,
        )[0]

    def test_distinct_roots_choose_left_for_obo_and_via(self):
        for plain_threshold in (None, 0.80):  # Primary and rules-only paths.
            with self.subTest(connector="OBO", plain_threshold=plain_threshold):
                decision = self.decide("OBO", 0.95, 0.90, plain_threshold)
                self.assertEqual((decision.decision, decision.verified_party_id, decision.connector_resolution),
                                 ("MATCH", "left", "OBO_LEFT"))
            with self.subTest(connector="VIA", plain_threshold=plain_threshold):
                decision = self.decide("VIA", 0.90, 0.95, plain_threshold)
                self.assertEqual((decision.decision, decision.verified_party_id, decision.connector_resolution),
                                 ("MATCH", "left", "VIA_LEFT"))

    def test_near_cutoff_guard_follows_new_preferred_side(self):
        for plain_threshold in (None, 0.80):
            for connector, left_score, right_score in (("OBO", 0.79, 0.95), ("VIA", 0.79, 0.95)):
                with self.subTest(connector=connector, plain_threshold=plain_threshold):
                    decision = self.decide(connector, left_score, right_score, plain_threshold)
                    self.assertEqual((decision.decision, decision.reason),
                                     ("NO_MATCH", "PREFERRED_SEGMENT_NEAR_CUTOFF"))

    def test_obo_uses_right_only_if_left_has_no_plausible_match(self):
        for plain_threshold in (None, 0.80):
            decision = self.decide("OBO", 0.40, 0.95, plain_threshold)
            self.assertEqual((decision.decision, decision.verified_party_id,
                              decision.connector_resolution), ("MATCH", "right", "ONLY_MATCH"))


if __name__ == "__main__":
    unittest.main()
