"""Mechanically refresh the standalone rules runtime from the POC source.

The generated files are checked in. Recipients do not run this script and do
not need the original repository. Keep the extraction list deliberately small:
no training, expansion, event emission, or reporting functions are copied.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "party_matching"
DEST = ROOT / "rules_only_handoff" / "party_matching_rules"


def extract(path: Path, names: list[str]) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    nodes = {getattr(node, "name", None): node for node in ast.parse("".join(lines)).body}
    chunks = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno, *(dec.lineno for dec in getattr(node, "decorator_list", []))])
        chunks.append("".join(lines[start - 1:node.end_lineno]).rstrip())
    return "\n\n\n".join(chunks) + "\n"


domain_names = [
    "normalize_name", "base_name", "compact_name", "PartyRecord",
    "OrganizationMention", "parse_mentions", "ExpansionCandidate",
    "MatchProposal", "FinalDecision",
]
domain_header = '''from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

CONNECTORS = re.compile(r"\\b(OBO|VIA)\\b", re.IGNORECASE)
LEGAL_SUFFIXES = {
    "ag", "bv", "co", "company", "corp", "corporation", "gmbh", "inc",
    "incorporated", "llc", "llp", "limited", "ltd", "nv", "plc", "pty",
    "sa", "sas", "sarl", "spa",
}

'''
matching_names = [
    "MentionRetriever", "collect_proposals", "_decide_records_legacy",
    "_decide_connectors", "decide_records", "pair_features", "target_features",
    "rule_identity_score", "char_similarity", "jaro_winkler_similarity",
    "levenshtein_similarity", "acronym_relation", "distinctive_token_conflict",
    "_combined_identity", "_best_per_root", "_match_method",
    "_rules_containment_evidence", "_decision_tier", "_guard_reason",
    "_no_match", "_token_idf", "_rss_mb",
]
matching_header = '''from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from .domain import (FinalDecision, LEGAL_SUFFIXES, MatchProposal,
                     OrganizationMention, PartyRecord, base_name, compact_name,
                     normalize_name, parse_mentions)
from .graph import VerifiedGraph

SKLEARN_ERROR = None
IDENTITY_FEATURES = [
    "exact", "base_exact", "compact_exact", "char_ratio", "jaro_winkler",
    "levenshtein", "token_jaccard", "raw_coverage", "candidate_coverage",
    "weighted_token_overlap", "weighted_candidate_coverage", "digit_conflict",
    "distinctive_token_conflict", "length_ratio", "char_tfidf", "word_tfidf",
    "rrf_score", "embedding_score", "official", "abbreviation", "collision_penalty",
    "acronym_relation", "legal_suffix_only_difference",
]
FEATURE_INDEX = {name: index for index, name in enumerate(IDENTITY_FEATURES)}

class CrossEncoderReranker:
    """No-op adapter: this handoff never loads or runs a cross-encoder."""
    def __init__(self, *args: Any) -> None:
        pass
    def score(self, proposals: list[MatchProposal]) -> None:
        pass

class FeatureScorer:
    """Exact POC rules formula, with no model artifact or learned inference."""
    def __init__(self, idf: dict[str, float], system_threshold: float,
                 plain_threshold: float) -> None:
        self.idf = idf
        self.system_threshold = system_threshold
        self.plain_threshold = plain_threshold
        self.timing: dict[str, float] = defaultdict(float)

    def score_proposals(self, proposals: list[MatchProposal], batch_size: int = 50_000) -> None:
        for start in range(0, len(proposals), max(1, batch_size)):
            batch = proposals[start:start + max(1, batch_size)]
            vectors = np.asarray([
                pair_features(item.mention_text, item.matched_name,
                              item.char_tfidf_score, item.word_tfidf_score,
                              item.rrf_score, item.embedding_score, item.candidate_type,
                              item.candidate_collision_count, self.idf)
                for item in batch
            ], dtype=np.float32)
            for item, vector in zip(batch, vectors):
                value = rule_identity_score(vector)
                if item.candidate_type != "official" and item.candidate_confidence is not None:
                    value = min(value, item.candidate_confidence)
                if item.exact and item.candidate_collision_count == 1:
                    floor = 0.999 if item.candidate_type == "official" else float(item.candidate_confidence or 0.0)
                    value = max(value, floor)
                item.feature_score = item.rules_score = value
                item.char_similarity = float(vector[FEATURE_INDEX["char_ratio"]])
                item.jaro_winkler = float(vector[FEATURE_INDEX["jaro_winkler"]])
                item.levenshtein = float(vector[FEATURE_INDEX["levenshtein"]])
                item.token_jaccard = float(vector[FEATURE_INDEX["token_jaccard"]])
                item.raw_coverage = float(vector[FEATURE_INDEX["raw_coverage"]])
                item.candidate_coverage = float(vector[FEATURE_INDEX["candidate_coverage"]])
                item.digit_conflict = bool(vector[FEATURE_INDEX["digit_conflict"]])
                item.distinctive_token_conflict = bool(vector[FEATURE_INDEX["distinctive_token_conflict"]])

    def score_target_vectors(self, vectors: list[list[float]], batch_size: int = 50_000) -> np.ndarray:
        if not vectors:
            return np.empty(0, dtype=np.float64)
        matrix = np.asarray(vectors, dtype=np.float32)
        margins = np.maximum(0.0, matrix[:, 0] - matrix[:, 3])
        return np.clip(matrix[:, 0] * (0.92 + np.minimum(0.08, margins)), 0.0, 0.999)

'''


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    matching_runtime = extract(SOURCE / "matching.py", matching_names)
    # Keep the lexical retrieval/scoring behavior, but remove the dormant
    # embedding loader/branch so the handoff cannot invoke an ML model.
    begin = matching_runtime.index("        self.embedding_model = None")
    end = matching_runtime.index("    def _parse", begin)
    matching_runtime = (
        matching_runtime[:begin]
        + "        self.embedding_model = None\n"
          "        self.embedding_matrix = None\n"
          "        self.ann_index = None\n"
          "        self.embedding_warning = None\n\n"
        + matching_runtime[end:]
    )
    begin = matching_runtime.index("    def _build_embeddings")
    end = matching_runtime.index("    def retrieve_top_candidates", begin)
    matching_runtime = matching_runtime[:begin] + matching_runtime[end:]
    begin = matching_runtime.index("        if self.embedding_model is not None and self.embedding_matrix is not None and queries:")
    end = matching_runtime.index("        for mention_index, hits in tuple(by_mention.items()):", begin)
    matching_runtime = matching_runtime[:begin] + matching_runtime[end:]
    (DEST / "domain.py").write_text(
        domain_header + extract(SOURCE / "domain.py", domain_names), encoding="utf-8"
    )
    (DEST / "matching.py").write_text(
        matching_header + matching_runtime, encoding="utf-8"
    )
    (DEST / "rules_enhancement.py").write_text(
        (SOURCE / "rules_enhancement.py").read_text(encoding="utf-8"), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
